"""RuleCOSI rule extractor.

This module contains the RuleCOSI extractor for classification problems.

The module structure is the following:

- The `BaseRuleCOSI` base class implements a common ``fit`` method
  for all the estimators in the module. This is done because in the future,
  the algorithm will work with regression problems as well.

- :class:`rulecosi.RuleCOSIClassifier` implements rule extraction from a
   variety of ensembles for classification problems.

"""

# Authors: Josue Obregon <jobregon@khu.ac.kr>
#
#
# License: TBD

import copy
import time
from abc import abstractmethod, ABCMeta
from ast import literal_eval
from math import sqrt
from operator import attrgetter

import numpy as np
import pandas as pd

import scipy.stats as st

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from rulecosi.rules import Rule, RuleSet
from .rule_extraction import RuleExtractorFactory
from .rule_heuristics import RuleHeuristics


def _ensemble_type(ensemble):
    """ Return the ensemble type

    :param ensemble:
    :return:
    """
    if isinstance(ensemble, (BaggingClassifier, RandomForestClassifier)):
        return 'bagging'
    elif isinstance(ensemble, GradientBoostingClassifier):
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'xgboost.sklearn.XGBClassifier '
                                      'ensembles you should install xgboost '
                                      'library.')
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'lightgbm.sklearn.LGBMClassifier'>":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'lightgbm.sklearn.LGBMClassifier '
                                      'ensembles you should install lightgbm '
                                      'library.')
        return 'gbt'
    elif str(
            ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'catboost.core.CatBoostClassifier '
                                      'ensembles you should install catboost '
                                      'library.')
        return 'gbt'
    else:
        raise NotImplementedError


def _statistical_error_estimate(N, e, z_alpha_half):
    """ Computes the statistical correction of the training error for
    estimating the generalization error.

    This function assumes that the errors on a rule (leaf node) follow a
    binomial distribution. Therefore, it computes the statistical correction
    as the upper limit of the normal approximation of a binomial distribution
    of the training error e.

    :param N: number of training records
    :param e: training error
    :param z_alpha_half: standardized value from a standard normal distribution
    :return: a float of the statistical correction of the training error e
    """
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(
        ((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N ** 2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator / denominator
    # 返回训练误差的统计纠正值，用于估计泛化误差。


class BaseRuleCOSI(BaseEstimator, metaclass=ABCMeta):
    """ Abstract base class for RuleCOSI estimators."""

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 percent_training=None,
                 early_stop=0,
                 metric='f1',
                 float_threshold=-1e-6,
                 column_names=None,
                 random_state=None,
                 verbose=0):
        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.percent_training = percent_training
        self.early_stop = early_stop
        self.metric = metric
        self.float_threshold = float_threshold
        self.column_names = column_names
        self.random_state = random_state
        self.verbose = verbose

    def _more_tags(self):
        return {'binary_only': True}

    def fit(self, X, y):
        """ Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.



        Returns
        -------
        self : object
        """
        pass

    @abstractmethod
    def _validate_and_create_base_ensemble(self):
        """ Validate the parameter of base ensemble and if it is None,
        it set the default ensemble GradientBoostingClassifier.

        """

        pass

    @abstractmethod
    def _remove_opposite_conditions(self, conditions, class_index):
        """ Removes conditions that have disjoint regions and will make a
        rule to be discarded because it would have null coverage.

            This function is used with the trees generated with CatBoost
            algorithm, which are called oblivious trees. This trees share the
            same splits among entire levels. So The rules generated when
            combining tend to create many rules with null coverage, so this
            function helps to avoid this problem and explore better the
            covered feature space combination.

        :param conditions: set of conditions' ids

        :param class_index: predicted class of the rule created with the set of
        conditions

        :return: set of conditions with no opposite conditions
        """

        pass

    @abstractmethod
    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the
        combination and simplification process This includes the set of good
        combinations G and bad combinations B, but can include other sets
        necessary to the combination process
        """
        pass

    @abstractmethod
    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different
        criteria

        :param ruleset: ruleset R to which the default rule will be added
        """
        pass

    @abstractmethod
    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using
        the procedure described in the paper [ref]

        :param ruleset1: ruleset 1 to be combined

        :param ruleset2: ruleset 2 to be combined

        :return: a new ruleset containing the result of the combination process
        """
        pass

    @abstractmethod
    def _sequential_covering_pruning(self, ruleset):
        """ Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it
        sorts it and find the best rule. Then the covered instances are
        removed from the training set and the process is repeated until one
        of three stopping criteria are met: 1. all the records of the
        training set are covered, 2. all the rules on ruleset are used or 3.
        there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        pass

    @abstractmethod
    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset using some loss metric.

        The function simplify the ruleset by iteratively removing conditions
        that minimize the pessimistic error. If all the conditions of a rule
        are removed, then the rule is discarded.

        :param ruleset:
        :return:
        """
        pass

    @abstractmethod
    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from
        each iteration cycle :param combined_rules: the combined rules that
        are obtained on each iteration of the combination cycle :return:the
        ruleset with best performance in accordance to the metric parameter

        比较两个规则集的性能，并返回性能最好的那一个。
            :param simplified_ruleset: 从每个迭代周期中传递的简化规则集
            :param combined_rules: 每个组合周期的组合规则集
            :return: 根据指定的度量参数返回性能最好的规则集
        """
        pass


class RuleCOSIClassifier(ClassifierMixin, BaseRuleCOSI):
    """ Tree ensemble Rule COmbiantion and SImplification algorithm for
    classification

    RuleCOSI extract, combines and simplify rules from a variety of tree
    ensembles and then constructs a single rule-based model that can be used
    for classification [1]. The ensemble is simpler and have a similar
    classification performance compared than that of the original ensemble.
    Currently only accept binary classification (March 2021)

    用于分类的树集成规则组合和简化算法
    RuleCOSI从各种树集成中提取、组合和简化规则，然后构建一个单一的基于规则的模型，可用于分类[1]。
    该集成模型相对于原始集成模型更简单，并且具有类似的分类性能。当前仅支持二元分类（2021年3月）

    Parameters
    ----------
    base_ensemble: BaseEnsemble object, default = None
        A BaseEnsemble estimator object. The supported types are:
       - :class:`sklearn.ensemble.RandomForestClassifier`
       - :class:`sklearn.ensemble.BaggingClassifier`
       - :class:`sklearn.ensemble.GradientBoostingClassifier`
       - :class:`xgboost.XGBClassifier`
       - :class:`catboost.CatBoostClassifier`
       - :class:`lightgbm.LGBMClassifier`

       If the estimator is already fitted, then the parameters n_estimators
       and max_depth used for fitting the ensemble are used for the combination
       process. If the estimator is not fitted, then the estimator will be
       first fitted using the provided parameters in the RuleCOSI object.
       Default value is None, which uses a
       :class:`sklearn.ensemble.GradientBoostingClassifier` ensemble.

        如果估计器已经拟合，则用于拟合集成模型的参数n_estimators和max_depth将用于组合过程。
        如果估计器尚未拟合，则将首先使用RuleCOSI对象中提供的参数来拟合估计器。
        默认值为None，使用:class:sklearn.ensemble.GradientBoostingClassifier集成模型。

    n_estimators: int, default=5
        The number of estimators used for fitting the ensemble,
        if it is not already fitted.
        如果集成模型尚未拟合，用于拟合集成模型的估计器数量。

    tree_max_depth: int, default=3
        The maximum depth of the individual tree estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    cov_threshold: float, default=0.0
        Coverage threshold of a rule to be considered for further combinations.
        (beta in the paper)The greater the value the more rules are discarded.
        Default value is 0.0, which it only discards rules with null coverage.
        要考虑进一步组合的规则的覆盖阈值。
        （论文中的beta）值越大，丢弃的规则越多。
        默认值为 0.0，它仅丢弃具有空覆盖率的规则。

    conf_threshold: float, default=0.5
        Confidence or rule accuracy threshold of a rule to be considered for
        further combinations (alpha in the paper). The greater the value, the
        more rules are discarded. Rules with high confidence are accurate rules.
        Default value is 0.5, which represents rules with higher than random
        guessing accuracy.
        要考虑进一步组合的规则的置信度或规则准确性阈值（本文中的 alpha）。
        值越大，丢弃的规则越多。 具有高置信度的规则是准确的规则。 默认值为0.5，表示比随机猜测精度更高的规则。

    c= float, 0.25
        Confidence level for estimating the upper bound of the statistical
        correction of the rule error. It is used for the generalization process.
        用于估计规则误差统计校正上限的置信水平。 它用于泛化过程。
        剪枝时，必须确保达到的最小置信度水平为 25%。学习到的模型对未知数据的预测能力


    percent_training= float, default=None
        Percentage of the training used for the combination and simplification
        process. If None, all the training data is usded. This is useful when
        the training data is too big because it helps to accelerate the
        simplification process. (experimental)
        用于组合和简化过程的培训百分比。
        如果没有，则使用所有训练数据。 当训练数据太大时，这很有用，因为它有助于加速简化过程。 （实验性）

    early_stop: float, default=0
        This parameter allows the algorithm to stop if a certain amount of
        iterations have passed without improving the metric. The amount is
        obtained from the truncated integer of n_estimators * ealry_stop.
        如果经过一定量的迭代而没有改进指标，则此参数允许算法停止。 该数量是从 n_estimators * ealry_stop 的截断整数中获得的。

    metric: string, default='f1'
        Metric that is optimized in the combination process. The default is
        f1. Other accepted measures are:
         - 'roc_auc' for AUC under the ROC curve
         - 'accuracy' for Accuracy
         在组合过程中优化的指标。 默认值为 f1。 其他可接受的措施有：
        - f1分数包含准确率（Precision）和召回率（Recall）
        - 'roc_auc' 表示 ROC 曲线下的 AUC
          ROC 曲线以假正例率（False Positive Rate，FPR）为横轴，真正例率（True Positive Rate，TPR）为纵轴，绘制了在不同阈值下的模型性能。
          AUC 等于 0.5 表示模型性能等同于随机猜测。
        - 准确度的“准确度”

    rule_order: string, default 'supp'
        Defines the way in the rules are ordered on each iteration. 'cov' order
        the rules first by coverage and 'conf' order the rules first by
        confidence or rule accuracy. 'supp' orders the rule by their support.
        This parameter affects the combination process and can be chosen
        conveniently depending on the desired results.
        定义规则在每次迭代中的排序方式。
        'cov'先按覆盖率排序规则，
        'conf'先按置信度或规则准确率排序规则。
        'supp'按支持度排序规则。
        此参数影响组合过程，并可以根据所需结果方便地选择。

    sort_by_class: bool or list, default None
        Define if the combined rules are ordered by class. If 'None', the rules
        are not forced to be ordered (although in some cases the result will
        be ordered). If 'True', the rules are ordered by class
        lexicographically. If a list is provided, the rules will be ordered by
        class following the order provided by the user. The list should
        contain the actual classes passed in y when calling the fit method.
        定义组合规则是否按类别排序。如果为“None”，则规则不必强制排序（尽管在某些情况下结果会被排序）。
        如果为“True”，则规则按字母顺序按类别排序。如果提供了列表，则规则将按照用户提供的顺序按类别排序。
        该列表应包含在调用fit方法时传递给y的实际类别。

    column_names: array of string, default=None
        Array of strings with the name of the columns in the data. This is
        useful for displaying the name of the features in the generated rules.
         包含数据中列名称的字符串数组。 这对于在生成的规则中显示功能的名称非常有用。

    random_state: int, RandomState instance or None, default=None
        Controls the random seed given to the ensembles when trained. RuleCOSI
        does not have any random process, so it affects only the ensemble
        training.
        控制训练时赋予集成模型的随机种子。RuleCOSI没有任何随机过程，因此仅影响集成模型的训练。

    verbose: int, default=0
        Controls the output of the algorithm during the combination process. It
         can have the following values:
        - 0 is silent
        - 1 output only the main stages of the algorithm
        - 2 output detailed information for each iteration
        控制组合过程中算法的输出。它可以具有以下值：
            0 表示静默
            1 仅输出算法的主要阶段
            2 输出每次迭代的详细信息

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    original_rulesets_ : array of RuleSet, shape (n_estimators,)
        The original rulesets extracted from the base ensemble.

    simplified_ruleset_ : RuleSet
        Combined and simplified ruleset extracted from the base ensemble.

    n_combinations_ : int
        Number of rule-level combinations performed by the algorithm.

    combination_time_ : float
        Time spent for the combination and simplification process

    ensemble_training_time_ : float
        Time spent for the ensemble training. If the ensemble was already
        trained, this is 0.


    References
    ----------
    .. [1] Obregon, J., Kim, A., & Jung, J. Y., "RuleCOSI: Combination and
           simplification of production rules from boosted decision trees for
           imbalanced classification", 2019.

    Examples
    --------
    # >>>：是一个提示符,通常用于示例代码或者在说明文档中展示代码示例时使用，用来表示用户应该在这里输入代码并执行。不会显示在调用函数中
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> from rulecosi import RuleCOSIClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
    ...                          n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    RuleCOSIClassifier(base_ensemble=GradientBoostingClassifier(),
                       n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.966...

    """

    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 c=0.25,
                 percent_training=None,
                 early_stop=0,
                 metric='f1',
                 rule_order='supp',
                 sort_by_class=None,
                 float_threshold=1e-6,
                 column_names=None,
                 random_state=None,
                 verbose=0
                 ):
        super().__init__(base_ensemble=base_ensemble,
                         n_estimators=n_estimators,
                         tree_max_depth=tree_max_depth,
                         percent_training=percent_training,
                         early_stop=early_stop,
                         metric=metric,
                         float_threshold=float_threshold,
                         column_names=column_names,
                         random_state=random_state,
                         verbose=verbose
                         )

        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.c = c
        self.rule_order = rule_order
        self.sort_by_class = sort_by_class

    def fit(self, X, y):
        """ Combine and simplify the decision trees from the base ensemble
        and builds a rule-based classifier using the training set (X,y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.



        Returns
        -------
        self : object
        """
        self._rule_extractor = None
        self._rule_heuristics = None
        self._base_ens_type = None
        self._weights = None
        self._global_condition_map = None
        self._bad_combinations = None
        self._good_combinations = None
        self._early_stop_cnt = 0
        self.alpha_half_ = None

        self.X_ = None
        self.y_ = None
        self.classes_ = None
        self.original_rulesets_ = None
        self.simplified_ruleset_ = None
        self.combination_time_ = None
        self.n_combinations_ = None
        self.ensemble_training_time_ = None

        # Check that X and y have correct shape
        if self.column_names is None:
            if isinstance(X, pd.DataFrame):
                self.column_names = X.columns
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.alpha_half_ = st.norm.ppf(1 - (self.c / 2))
        # st.norm.ppf是 SciPy 库中正态分布的累积分布函数的逆函数，用于计算给定概率的对应分位点（即给定概率下的标准正态分布的值）。
        # 将置信水平 self.c 转换为对应的正态分布的分位数
        # 计算对应于置信水平 1 - (self.c / 2) 的正态分布分位数

        if self.percent_training is None:
            self.X_ = X
            self.y_ = y
        else:
            x, _, y, _ = train_test_split(X, y,
                                          test_size=(1 - self.percent_training),
                                          shuffle=True, stratify=y,
                                          random_state=self.random_state)
            self.X_ = x
            self.y_ = y

        # add rule ordering funcionality (2023.03.30)
        self._sorting_list = ['cov', 'conf', 'supp']
        self._sorting_list.remove(self.rule_order)
        self._sorting_list.insert(0, self.rule_order)
        self._sorting_list.reverse()
        if self.sort_by_class is not None:
            if isinstance(self.sort_by_class, bool):
                self.sort_by_class = self.classes_.tolist()

        if self.n_estimators is None or self.n_estimators < 2:
            raise ValueError(
                "Parameter n_estimators should be at "
                "least 2 for using the RuleCOSI method.")

        if self.verbose > 0:
            print('Validating original ensemble...')
        try:
            if self.base_ensemble is None:
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble_ = self.base_ensemble
            self._base_ens_type = _ensemble_type(self.base_ensemble_)
        except NotImplementedError:
            print(
                f'Base ensemble of type {type(self.base_ensemble_).__name__} '
                f'is not supported.')
        try:
            check_is_fitted(self.base_ensemble_)
            self.ensemble_training_time_ = 0
            if self.verbose > 0:
                print(
                    f'{type(self.base_ensemble_).__name__} already trained, '
                    f'ignoring n_estimators and '
                    f'tree_max_depth parameters.')
        except NotFittedError:
            self.base_ensemble_ = self._validate_and_create_base_ensemble()
            if self.verbose > 0:
                print(
                    f'Training {type(self.base_ensemble_).__name__} '
                    f'base ensemble...')
            start_time = time.time()
            self.base_ensemble_.fit(X, y)
            end_time = time.time()
            self.ensemble_training_time_ = end_time - start_time
            if self.verbose > 0:
                print(
                    f'Finish training {type(self.base_ensemble_).__name__} '
                    f'base ensemble'
                    f' in {self.ensemble_training_time_} seconds.')

        start_time = time.time()

        # First step is extract the rules
        self._rule_extractor = RuleExtractorFactory.get_rule_extractor(
            self.base_ensemble_, self.column_names,
            self.classes_, self.X_, self.y_, self.float_threshold)
        if self.verbose > 0:
            print(
                f'Extracting rules from {type(self.base_ensemble_).__name__} '
                f'base ensemble...')
        self.original_rulesets_, \
        self._global_condition_map = self._rule_extractor.extract_rules()  # 原始规则的提取
        # 创建对象的深度拷贝。深度拷贝意味着会递归地拷贝对象及其内部包含的所有对象，而不是仅仅拷贝对象本身。
        # 这样做的目的是防止在处理规则集合时修改原始规则集合。
        processed_rulesets = copy.deepcopy(self.original_rulesets_)
        # self.processed_rulesets =processed_rulesets

        # We create the heuristics object which will compute all the
        # heuristics related measures
        # 我们创建启发式对象，它将计算所有与启发式相关的度量--根据规则准确度和规则覆盖率按降序排列
        # 选择精度大，找到具有良好准确性和覆盖率的规则，通过置信度删除规则，直到规则全部覆盖
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=
                                               self._global_condition_map,
                                               cov_threshold=self.cov_threshold,
                                               conf_threshold=
                                               self.conf_threshold)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()

        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            for ruleset in processed_rulesets:
                for rule in ruleset:
                    new_A = self._remove_opposite_conditions(set(rule.A),
                                                             rule.class_index)
                    rule.A = new_A
        # self.processed_rulesets =processed_rulesets
        for ruleset in processed_rulesets:
            self._rule_heuristics.compute_rule_heuristics(
                ruleset, recompute=True)

        self.simplified_ruleset_ = processed_rulesets[0]

        self._simplify_rulesets(
            self.simplified_ruleset_)
        y_pred = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.compute_class_perf_fast(y_pred,
                                                         self.y_,
                                                         self.metric)
        self.simplified_ruleset_.rules.pop()  # 用于移除并返回列表中的最后一个元素。

        # 启发式搜索顺序剪枝的结果
        self.simplified = self.simplified_ruleset_.rules

        self.n_combinations_ = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)

        if self.verbose > 0:
            print(f'Start combination process...')
            if self.verbose > 1:
                print(
                    f'Iteration {0}, Rule size: '
                    f'{len(self.simplified_ruleset_.rules)}, '
                    f'{self.metric}: '
                    f'{self.simplified_ruleset_.metric(self.metric)}')
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset_,
                                                    processed_rulesets[i])

            if self.verbose > 1:
                print(f'Iteration{i}:')
                print(
                    f'\tCombined rules size: {len(combined_rules.rules)} rules')
            # prune inaccurate rules
            self._sequential_covering_pruning(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSequential covering pruned rules size: '
                    f'{len(combined_rules.rules)} rules')
            # simplify rules
            self._simplify_rulesets(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSimplified rules size: '
                    f'{len(combined_rules.rules)} rules')

            # skip if the combined rules are empty
            if len(combined_rules.rules) == 0:
                if self.verbose > 1:
                    print(f'\tCombined rules are empty, skipping iteration.')
                continue

            self.simplified_ruleset_, best_ruleset = self._evaluate_combinations(
                self.simplified_ruleset_, combined_rules)

            self.rrrr = self.simplified_ruleset_  # ---结果输出
            self.combined_rules = combined_rules
            # self.best_ruleset = best_ruleset

            if self._early_stop_cnt >= early_stop:
                break
            if self.simplified_ruleset_.metric() == 1:
                break

        self.simplified_ruleset_.rules[:] = [rule for rule in
                                             self.simplified_ruleset_.rules
                                             if rule.cov > 0]
        if self.verbose > 0:
            print(f'Finish combination process, adding default rule...')

        _ = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.prune_condition_map()
        end_time = time.time()
        self.combination_time_ = end_time - start_time
        if self.verbose > 0:
            print(
                f'R size: {len(self.simplified_ruleset_.rules)}, {self.metric}:'
                f' {self.simplified_ruleset_.metric(self.metric)}')

        return self

    def _more_tags(self):
        return {'binary_only': True}

    def predict(self, X):
        """ Predict classes for X.

        The predicted class of an input sample. The prediction use the
        simplified ruleset and evaluate the rules one by one. When a rule
        covers a sample, the head of the rule is returned as predicted class.
        输入样本的预测类别。 预测使用简化规则集并一一评估规则。 当规则覆盖一个样本，规则的头部作为预测类返回。

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class. The class with the highest value in the class
            distribution of the fired rule.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)  # 对输入的数据 X 进行验证和转换，确保其符合模型的要求。
        # 检查已经拟合过的模型对象 self 中保存的训练数据 self.X_ 的特征数量是否与当前输入数据 X 的特征数量相匹配。
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleCOSIClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )

        return self.simplified_ruleset_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is obtained from
        the class distribution of the fired rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleCOSIClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )
        return self.simplified_ruleset_.predict_proba(X)

    def _initialize_sets(self):
        """ Initialize the sets that are going to be used during the
        combination and simplification process This includes the set of good
        combinations G and bad combinations B. It also includes the bitsets
        for the training data as well as the bitsets for each of the conditions
        """
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._rule_heuristics.initialize_sets()

    def _validate_and_create_base_ensemble(self):
        """ Validate the parameter of base ensemble and if it is None,
        it set the default ensemble GradientBoostingClassifier.

        """

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))
        if self.base_ensemble is None:
            if is_classifier(self):
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            elif is_regressor(self):
                self.base_ensemble_ = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                raise ValueError(
                    "You should choose an original classifier/regressor "
                    "ensemble to use RuleCOSI method.")
        self.base_ensemble_.n_estimators = self.n_estimators
        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            self.base_ensemble_.set_params(n_estimators=self.n_estimators,
                                           depth=self.tree_max_depth)
        elif isinstance(self.base_ensemble_, BaggingClassifier):
            if is_classifier(self):
                self.base_ensemble_.base_estimator = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble_.base_estimator = DecisionTreeRegressor(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
        else:
            self.base_ensemble_.max_depth = self.tree_max_depth
        return clone(self.base_ensemble_)

    def _remove_opposite_conditions(self, conditions, class_index):
        """ Removes conditions that have disjoint regions and will make a
        rule to be discarded because it would have null coverage.

            This function is used with the trees generated with CatBoost
            algorithm, which are called oblivious trees. This trees share the
            same splits among entire levels. So The rules generated when
            combining tend to create many rules with null coverage, so this
            function helps to avoid this problem and explore better the
            covered feature space combination.

        :param conditions: set of conditions' ids

        :param class_index: predicted class of the rule created with the set of
        conditions

        :return: set of conditions with no opposite conditions
        """

        att_op_list = [[cond[1].att_index,
                        cond[1].op.__name__, cond[0]]
                       for cond in conditions]

        att_op_list = np.array(att_op_list, dtype=object)
        att_op_list = att_op_list[att_op_list[:, 0].argsort()]

        # Second part is to remove opposite operator
        # conditions (e.g. att1>=5  att1<5)
        # create list for removing opposite conditions
        dict_opp_cond = {
            i[0]: att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 2]
            for i in att_op_list}
        # create generator to traverse just conditions with the same att_index
        # and different operator that appear
        # more than once
        gen_opp_cond = ((att, conds) for (att, conds) in dict_opp_cond.items()
                        if len(conds) > 1)
        for (_, conds) in gen_opp_cond:
            list_conds = [(int(id_),
                           self._rule_heuristics.get_conditions_heuristics(
                               {(int(id_),
                                 self._global_condition_map[int(id_)])})[0][
                               'supp'][class_index])
                          for id_
                          in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(
                best_condition)  # remove the edge condition of the box from
            # the list so it will remain
            [conditions.remove((cond[0], self._global_condition_map[cond[0]]))
             for cond in list_conds]
        return frozenset(conditions)

    def _sort_ruleset(self, ruleset):
        """ Sort the ruleset in place according to the rule_order parameter.

        :param ruleset: ruleset to be ordered
        """
        if len(ruleset.rules) == 0:
            return

        # 按 rule.A 属性的长度（从长到短）进行排序。如果长度相同，则按规则的字符串表示进行排序。
        ruleset.rules.sort(key=lambda rule: (-1 * len(rule.A), rule.str))
        for attr in self._sorting_list:  # self._sorting_list:['cov', 'conf', 'supp']
            ruleset.rules.sort(key=attrgetter(attr), reverse=True)  # attrgetter(attr) 按每个属性从大到小排序。

        if self.sort_by_class is not None:
            ruleset.rules.sort(key=lambda rule: self.sort_by_class.index(rule.y.item()))  # 类的顺序对规则进行排序
        # if self.rule_order == 'cov':
        #     ruleset.rules.sort(key=lambda rule: (rule.cov, rule.conf, rule.supp,
        #                                          -1 * len(rule.A),
        #                                          rule.str), reverse=True)
        #
        # elif self.rule_order == 'conf':
        #     ruleset.rules.sort(key=lambda rule: (rule.conf, rule.cov, rule.supp,
        #                                          -1 * len(rule.A),
        #                                          rule.str), reverse=True)
        # elif self.rule_order == 'supp':
        #     ruleset.rules.sort(key=lambda rule: (rule.supp, rule.conf, rule.cov,
        #                                          -1 * len(rule.A),
        #                                          rule.str), reverse=True)

    def _combine_rulesets(self, ruleset1, ruleset2):
        """ Combine all the rules belonging to ruleset1 and ruleset2 using
        the procedure described in the paper [ref]

        Main guiding procedure for combining rulesets for classification,
        make a combination of each of the class with itself and all the other
        classes

        :param ruleset1: First ruleset to be combined

        :param ruleset2: Second ruleset to be combiuned

        :return: ruleset containing the combination of ruleset1 and ruleset 2
        """
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if
                              (rule1.y == [class_one])]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if
                              (rule2.y == [class_two])]
                combined_rules.update(
                    self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        combined_rules = RuleSet(list(combined_rules),
                                 self._global_condition_map,
                                 classes=self.classes_)
        self._sort_ruleset(combined_rules)
        return combined_rules

    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        """ Actual combination procedure between to class-sliced rulesets

        :param s_ruleset1: sliced ruleset 1 according to a class

        :param s_ruleset2: sliced ruleset according to a class

        :return: a set of rules containing the combined rules of s_ruleset1 and
         s_ruleset2
        """
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A) == 0 or len(r2.A) == 0:
                    continue
                heuristics_dict = self._rule_heuristics.combine_heuristics(
                    r1.heuristics_dict, r2.heuristics_dict)

                r1_AUr2_A = set(r1.A.union(r2.A))

                if heuristics_dict['cov'] == 0:
                    self._bad_combinations.add(frozenset(r1_AUr2_A))
                    continue

                if frozenset(r1_AUr2_A) in self._bad_combinations:
                    continue

                self.n_combinations_ += 1  # count the actual
                # number of combinations

                # create the new rule and compute class distribution
                # and predicted class
                weight = None

                if self._weights is None:
                    ens_class_dist = np.mean(
                        [r1.ens_class_dist, r2.ens_class_dist],
                        axis=0).reshape(
                        (len(self.classes_),))
                else:
                    ens_class_dist = np.average(
                        [r1.ens_class_dist, r2.ens_class_dist],
                        axis=0,
                        weights=[r1.weight,
                                 r2.weight]).reshape(
                        (len(self.classes_),))
                    weight = (r1.weight() + r2.weight) / 2
                logit_score = 0

                class_dist = ens_class_dist
                y_class_index = np.argmax(class_dist).item()
                y = np.array([self.classes_[y_class_index]])

                new_rule = Rule(frozenset(r1_AUr2_A),
                                class_dist=class_dist,
                                ens_class_dist=ens_class_dist,
                                local_class_dist=ens_class_dist,
                                # rule_class_dist,
                                logit_score=logit_score, y=y,
                                y_class_index=y_class_index,
                                classes=self.classes_, weight=weight)

                if new_rule in self._good_combinations:
                    heuristics_dict = self._good_combinations[new_rule]
                    new_rule.set_heuristics(heuristics_dict)
                    combined_rules.add(new_rule)
                else:
                    new_rule.set_heuristics(heuristics_dict)
                    if new_rule.conf > self.conf_threshold and \
                            new_rule.cov > self.cov_threshold:
                        combined_rules.add(new_rule)
                        self._good_combinations[new_rule] = heuristics_dict
                    else:
                        self._bad_combinations.add(frozenset(r1_AUr2_A))

        return combined_rules

    def _simplify_conditions(self, conditions):
        """ Remove redundant conditions of a single rule.

        Redundant conditions are conditions with the same attribute and same
        operator but different value.

        For example: (att1 > 5) ^ (att1 > 10). In this case we keep the
        second one because it contains the first one

        :param conditions: set of conditions' ids

        :return: return the set of conditions with no redundancy
        """
        cond_map = self._global_condition_map  # just for readability
        # create list with this format ["(att_index, 'OPERATOR')", 'cond_id']
        att_op_list = [
            [(cond[1].att_index, cond[1].op.__name__), cond[0]]
            for cond in conditions]
        att_op_list = np.array(att_op_list, dtype=object)
        # First part is to remove redundant conditions
        # (e.g. att1>5 and att1> 10)
        # create list for removing redundant conditions
        dict_red_cond = {
            i[0]: att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 1]
            for i in att_op_list}
        # create generator to traverse just conditions with the same att_index
        # and operator that appear more than once
        gen_red_cond = ((att_op, conds) for (att_op, conds) in
                        dict_red_cond.items() if len(conds) > 1)

        for (att_op, conds) in gen_red_cond:
            tup_att_op = literal_eval(att_op)
            list_conds = {cond_map[int(id_)] for id_ in conds}
            if tup_att_op[1] in ['lt', 'le']:
                edge_condition = min(list_conds, key=lambda
                    item: item.value)  # condition at the edge of the box
            if tup_att_op[1] in ['gt', 'ge']:
                edge_condition = max(list_conds, key=lambda item: item.value)
            list_conds.remove(
                edge_condition)  # remove the edge condition of the box from
            # the list, so it will remain
            # [conditions.remove(hash(cond)) for cond in list_conds]
            [conditions.remove((hash(cond), cond)) for cond in list_conds]

        return frozenset(conditions)

    def _sequential_covering_pruning(self, ruleset):
        """Reduce the size of the ruleset by removing meaningless rules.

        The function first, compute the heuristic of the ruleset, then it
        sorts it and find the best rule. Then the covered instances are
        removed from the training set and the process is repeated until one
        of three stopping criteria are met: 1. all the records of the
        training set are covered, 2. all the rules on ruleset are used or 3.
        there is no rule that satisfies the coverage and accuracy constraints.

        :param ruleset: ruleset R to be pruned
        :return:
        """
        return_ruleset = []
        not_cov_samples = self._rule_heuristics.ones
        found_rule = True
        while len(
                ruleset.rules) > 0 and \
                self._rule_heuristics.bitarray_.get_number_ones(
                    not_cov_samples) > 0 \
                and found_rule:
            self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                          not_cov_samples)
            self._sort_ruleset(ruleset)
            found_rule = False
            for rule in ruleset:
                result, \
                not_cov_samples = self._rule_heuristics.rule_is_accurate(
                    rule,
                    not_cov_samples=not_cov_samples)
                if result:
                    return_ruleset.append(rule)
                    ruleset.rules.remove(rule)
                    found_rule = True
                    break
        ruleset.rules[:] = return_ruleset

    def _simplify_rulesets(self, ruleset):
        """Simplifies the ruleset inplace using the pessimist error.

        The function simplify the ruleset by iteratively removing conditions
        that minimize the pessimistic error. If all the conditions of a rule
        are removed, then the rule is discarded.
        该函数通过迭代地删除最小化悲观错误的条件来简化规则集。如果一个规则的所有条件都被移除，则该规则将被丢弃。

        1.对于规则集中的每个规则：
        使用 _simplify_conditions 方法简化规则的条件。
        更新规则的字符串表示形式。
        计算规则的基线误差（base_line_error）作为简化的起点。
        如果基线误差大于0且规则条件集非空，则进入循环，尝试移除一个条件以减小误差。
            对于每个条件，计算移除该条件后的悲观误差，并记录下来。
            选择悲观误差最小的条件进行移除。
            更新基线误差。
        最终，更新规则集的启发式信息（heuristics）。

        2.最后，根据一些条件过滤规则集：
            移除条件为空的规则。
            移除覆盖率（cov）低于阈值的规则。
            移除置信度（conf）低于阈值的规则。

        3.重新计算规则集的启发式信息。

        4.对规则集进行排序。

        :param ruleset:
        """
        for rule in ruleset:
            rule.A = self._simplify_conditions(set(rule.A))
            rule.update_string_representation()
            base_line_error = self._compute_pessimistic_error(rule.A,
                                                              rule.class_index)
            min_error = 0
            # c = 25%--base_line_error使用训练记录的数量、训练误差、标准化值
            # 标准化值：使用c形成标准正态分布，计算在标准正态分布下累积概率为 1 - (self.c / 2) 的点的逆函数值（即 1 - (self.c / 2) 累积概率对应的分位数）
            while min_error <= base_line_error and len(rule.A) > 0 \
                    and base_line_error > 0:
                errors = [(cond,
                           self._compute_pessimistic_error(
                               rule.A.difference({cond}), rule.class_index),
                           cond[1].str)
                          for cond
                          in rule.A]

                min_error_tup = min(errors, key=lambda tup: (tup[1],
                                                             tup[2]))

                min_error = min_error_tup[1]
                if min_error <= base_line_error:
                    base_line_error = min_error
                    min_error = 0
                    rule_conds = set(rule.A)
                    rule_conds.remove(min_error_tup[0])
                    rule.A = frozenset(rule_conds)
                    rule.update_string_representation()

        self._rule_heuristics.compute_rule_heuristics(ruleset, recompute=True)

        ruleset.rules[:] = [rule for rule in ruleset
                            if 0 < len(rule.A)
                            and rule.cov > self.cov_threshold
                            and rule.conf > self.conf_threshold
                            ]

        self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                      sequential_covering=True)
        self._sort_ruleset(ruleset)

    def _compute_pessimistic_error(self, conditions, class_index,
                                   not_cov_samples=None):
        """ Computes a statistical correction to the training error to
        estimate the generalization error of one rule.

        This function assumes that the errors on the rule follow a binomial
        distribution. Therefore, it computes the statistical correction as
        the upper limit of the normal approximation of a binomial
        distribution of the training error e.

        :param conditions: set of conditions' ids

        :param class_index: predicted class index of the rule

        :return: the statistical correction of the training error of that
        rule (between 0 and 100)
        """

        if len(conditions) == 0:
            e = (self.X_.shape[
                     0] - self._rule_heuristics.bitarray_.get_number_ones(
                self._rule_heuristics.training_bit_sets[
                    class_index])) / self.X_.shape[0]
            return 100 * _statistical_error_estimate(self.X_.shape[0], e,
                                                     self.alpha_half_)

        heuristics_dict, _ = self._rule_heuristics.get_conditions_heuristics(
            conditions, not_cov_mask=not_cov_samples)

        total_instances = heuristics_dict['cov_count']
        accurate_instances = heuristics_dict['class_cov_count'][class_index]
        error_instances = total_instances - accurate_instances

        if total_instances == 0:
            return 0
        else:
            e = error_instances / total_instances  # totalInstances

        return 100 * _statistical_error_estimate(total_instances, e,
                                                 self.alpha_half_)
        # 该方法基于假设规则的错误遵循二项分布，因此它通过计算规则训练误差的正态近似的上限来进行统计校正。
        # _statistical_error_estimate 函数来进行正态分布近似的统计误差估计。

    def _add_default_rule(self, ruleset):
        """ Add a default rule at the end of the ruleset depending different
        criteria

        :param ruleset: ruleset R to which the default rule will be added
        """

        predictions, covered_instances = ruleset._predict(self.X_)
        not_cov_samples = ~covered_instances

        all_covered = False
        if not_cov_samples.sum() == 0:
            uncovered_dist = np.array(
                [self._rule_heuristics.bitarray_.get_number_ones(
                    self._rule_heuristics.training_bit_sets[i]) for i in
                    range(len(self.classes_))])
            all_covered = True
        else:
            uncovered_labels = self.y_[not_cov_samples]
            uncovered_dist = np.array(
                [(uncovered_labels == class_).sum() for class_ in
                 self.classes_])

        default_class_idx = np.argmax(uncovered_dist)
        predictions[not_cov_samples] = self.classes_[default_class_idx]
        default_rule = Rule({},
                            class_dist=uncovered_dist / uncovered_dist.sum(),
                            y=np.array([self.classes_[default_class_idx]]),
                            y_class_index=default_class_idx,
                            classes=self.classes_, n_samples=uncovered_dist)
        if not all_covered:
            default_rule.cov = not_cov_samples.sum() / self.X_.shape[0]
            default_rule.conf = uncovered_dist[
                                    default_class_idx] / not_cov_samples.sum()
            default_rule.supp = uncovered_dist[default_class_idx] / \
                                self.X_.shape[0]
        ruleset.rules.append(default_rule)

        return predictions

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        """ Compare the performance of two rulesets and return the best one

        :param simplified_ruleset: the simplified rules that are carried from
        each iteration cycle

        :param combined_rules: the combined rules that are obtained on each
        iteration of the combination cycle

        :return:the ruleset with best performance in accordance to the metric
        parameter

        比较两个规则集的性能并返回最佳规则集

        :p aram simplified_ruleset：从
        每个迭代周期

        :p aram combined_rules：在每个
        组合循环的迭代

        ：return：根据指标具有最佳性能的规则集
        参数
        """
        y_pred = self._add_default_rule(combined_rules)

        combined_rules.compute_class_perf_fast(y_pred, self.y_,
                                               self.metric)

        # if rule_added:
        combined_rules.rules.pop()

        if combined_rules.metric(self.metric) > simplified_ruleset.metric(
                self.metric):
            self._early_stop_cnt = 0
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Combined rules: '
                    f'{combined_rules.metric(self.metric)}')
            return combined_rules, 'comb'
        else:
            self._early_stop_cnt += 1
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Previous combined rules: '
                    f'{simplified_ruleset.metric(self.metric)}')
            return simplified_ruleset, 'simp'
