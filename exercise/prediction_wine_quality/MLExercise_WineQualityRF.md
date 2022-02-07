# Wine Quality prediction using Decision Forest
Classification exercise using TensorFlow

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)

[Exercise example: Ranking Conditions](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab#installing_tensorflow_decision_forests)

[Exercise example: Building a soft decision tree, from scratch](https://towardsdatascience.com/building-a-decision-tree-in-tensorflow-742438cb483e)

[Exercise example: K-means cluster, from scratch](https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25)


```python
!pip install tensorflow_decision_forests -q
!pip install shap -q
import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import math
```

    [K     |████████████████████████████████| 6.3 MB 4.7 MB/s 
    [K     |████████████████████████████████| 356 kB 5.2 MB/s 
    [?25h  Building wheel for shap (setup.py) ... [?25l[?25hdone
    


```python
!pip install wurlitzer -q
try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes

from IPython.core.magic import register_line_magic
from IPython.display import Javascript
```


```python
!pip install bokeh -q
import scipy.special

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.resources import INLINE
import bokeh.io

bokeh.io.output_notebook(INLINE)
```


```python
def make_plot(title, hist, edges, x, pdf, cdf):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
    p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    return p

def plot_distribution_check(vec_review, str_name):

    hist, edges = np.histogram(vec_review, density=True, bins=50)

    x_min = min(vec_review)
    x_max = max(vec_review)
    x_mu  = np.mean(vec_review)
    x_sig = np.std(vec_review)

    x = np.linspace(x_min, x_max, 1000)
    pdf = 1/(x_sig * np.sqrt(2*np.pi)) * np.exp(-(x-x_mu)**2 / (2*x_sig**2))
    cdf = (1+scipy.special.erf((x-x_mu)/np.sqrt(2*x_sig**2)))/2

    plt_obj = make_plot(f"DIstribution checks for: {str_name}", hist, edges, x, pdf, cdf)

    return plt_obj
```

## Import and process data

Wine quality data provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)


```python
# Wine data
str_df_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
str_df_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

df_red = pd.read_csv(str_df_red, sep = ';')
df_wht = pd.read_csv(str_df_white, sep = ';')
```


```python
df_red.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Histogram plots
str_check = df_red.columns.to_list();
vec_plot = []

for iter in str_check:
  vec_plot.append(plot_distribution_check(df_red[iter], iter))
show(gridplot(vec_plot, ncols=3, plot_width =400, plot_height=400, toolbar_location=None))
```










<div class="bk-root" id="d2737548-db33-4ffc-b452-34bcf989d21e" data-root-id="1914"></div>





Some data transformations are recommended:
* alcohol - log scale
* free sulfer dioxide - log scale
* total sulfer dioxide - log scale


```python
df_red['alcohol'] = np.log(df_red['alcohol'])
df_red['free sulfur dioxide'] = np.log(df_red['free sulfur dioxide'])
df_red['total sulfur dioxide'] = np.log(df_red['total sulfur dioxide'] )

# Re-check the variables
# Histogram plots
str_check = df_red.columns.to_list();
vec_plot = []

for iter in str_check:
  vec_plot.append(plot_distribution_check(df_red[iter], iter))
show(gridplot(vec_plot, ncols=3, plot_width =400, plot_height=400, toolbar_location=None))
```










<div class="bk-root" id="8b2bf02b-3573-4d09-96d4-aa3b6f96da5c" data-root-id="3763"></div>






```python
df_wht.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_red.loc[:,'quality'] = df_red['quality'].map(str)
classes = sorted(df_red['quality'].unique().tolist())
print(classes)

df_red['quality'] = df_red.quality.map(classes.index)
```

    ['3', '4', '5', '6', '7', '8']
    


```python
#df_red['quality_bool'] = ['good' if iter > 5 else 'bad' for iter in df_red.quality]
#df_wht['quality_bool'] = ['good' if iter > 5 else 'bad' for iter in df_wht.quality]
```

Function to split test and train


```python
# Split the dataset into a training and a testing dataset.

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_red, test_ds_red = split_dataset(df_red)
print(f"Red Wine: {len(train_ds_red)} examples in training, {len(test_ds_red)} examples for testing.")
train_ds_white, test_ds_white = split_dataset(df_wht)
print(f"White Wine: {len(train_ds_white)} examples in training, {len(test_ds_white)} examples for testing.")
```

    Red Wine: 1131 examples in training, 468 examples for testing.
    White Wine: 3438 examples in training, 1460 examples for testing.
    

## Red wine prediction


```python
# Variable of interest 
label = 'quality'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_red, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_red, label=label)
```

    WARNING:absl:Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    WARNING:absl:Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    


```python
def vec_prediction(predict_prob):
  
  n, n_prob = predict_prob.shape

  vec_pred_class = [0] * n
  vec_pred_class_idx = [0] * n
  vec_pred_unc  = [0] * n

  for iter in range(0, n):
    vec_prob = predict_prob[iter, :]
    # Most likely based on maximum of probability
    
    idx_likely = list(vec_prob).index(max(vec_prob))
    #   save the index, for comparison reason
    vec_pred_class_idx[iter] = idx_likely

    # save the prediction
    vec_pred_class[iter] = classes[idx_likely]
    
    # Save standard error of prediction probability
    vec_pred_unc[iter] = np.std(vec_prob)

  return vec_pred_class_idx, vec_pred_class, vec_pred_unc
```

### What's up with the __LABEL?




```python
#%set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel()

# Optionally, add evaluation metrics.
model_1.compile(metrics=["accuracy"])

# Train the model.
# "sys_pipes" is optional. It enables the display of the training logs.
with sys_pipes():
  model_1.fit(x=train_ds)
```

    2021-09-24 19:32:07.557751: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
    

    18/18 [==============================] - 5s 2ms/step
    

    [INFO kernel.cc:746] Start Yggdrasil model training
    [INFO kernel.cc:747] Collect training examples
    [INFO kernel.cc:392] Number of batches: 18
    [INFO kernel.cc:393] Number of examples: 1131
    [INFO kernel.cc:769] Dataset:
    Number of records: 1131
    Number of columns: 12
    
    Number of columns by type:
    	NUMERICAL: 11 (91.6667%)
    	CATEGORICAL: 1 (8.33333%)
    
    Columns:
    
    NUMERICAL: 11 (91.6667%)
    	0: "alcohol" NUMERICAL mean:2.33819 min:2.12823 max:2.70136 sd:0.0987049
    	1: "chlorides" NUMERICAL mean:0.0888099 min:0.012 max:0.611 sd:0.052287
    	2: "citric_acid" NUMERICAL mean:0.269461 min:0 max:1 sd:0.197325
    	3: "density" NUMERICAL mean:0.996752 min:0.99007 max:1.00369 sd:0.00189263
    	4: "fixed_acidity" NUMERICAL mean:8.34288 min:4.6 max:15.9 sd:1.75203
    	5: "free_sulfur_dioxide" NUMERICAL mean:2.52032 min:0 max:4.21951 sd:0.689124
    	6: "pH" NUMERICAL mean:3.31007 min:2.74 max:4.01 sd:0.154555
    	7: "residual_sugar" NUMERICAL mean:2.50385 min:0.9 max:15.4 sd:1.33419
    	8: "sulphates" NUMERICAL mean:0.66015 min:0.33 max:2 sd:0.173427
    	9: "total_sulfur_dioxide" NUMERICAL mean:3.57523 min:1.79176 max:5.66643 sd:0.711783
    	10: "volatile_acidity" NUMERICAL mean:0.527798 min:0.12 max:1.58 sd:0.181832
    
    CATEGORICAL: 1 (8.33333%)
    	11: "__LABEL" CATEGORICAL integerized vocab-size:7 no-ood-item
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO kernel.cc:772] Configure learner
    [INFO kernel.cc:797] Training config:
    learner: "RANDOM_FOREST"
    features: "alcohol"
    features: "chlorides"
    features: "citric_acid"
    features: "density"
    features: "fixed_acidity"
    features: "free_sulfur_dioxide"
    features: "pH"
    features: "residual_sugar"
    features: "sulphates"
    features: "total_sulfur_dioxide"
    features: "volatile_acidity"
    label: "__LABEL"
    task: CLASSIFICATION
    [yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
      num_trees: 300
      decision_tree {
        max_depth: 16
        min_examples: 5
        in_split_min_examples_check: true
        missing_value_policy: GLOBAL_IMPUTATION
        allow_na_conditions: false
        categorical_set_greedy_forward {
          sampling: 0.1
          max_num_items: -1
          min_item_frequency: 1
        }
        growing_strategy_local {
        }
        categorical {
          cart {
          }
        }
        num_candidate_attributes_ratio: -1
        axis_aligned_split {
        }
        internal {
          sorting_strategy: PRESORTED
        }
      }
      winner_take_all_inference: true
      compute_oob_performances: true
      compute_oob_variable_importances: false
      adapt_bootstrap_size_ratio_for_maximum_training_duration: false
    }
    
    [INFO kernel.cc:800] Deployment config:
    num_threads: 6
    
    [INFO kernel.cc:837] Train model
    [INFO random_forest.cc:303] Training random forest on 1131 example(s) and 11 feature(s).
    [INFO random_forest.cc:578] Training of tree  1/300 (tree index:0) done accuracy:0.512136 logloss:17.5844
    [INFO random_forest.cc:578] Training of tree  11/300 (tree index:12) done accuracy:0.608541 logloss:6.10203
    [INFO random_forest.cc:578] Training of tree  21/300 (tree index:24) done accuracy:0.627763 logloss:3.57887
    [INFO random_forest.cc:578] Training of tree  31/300 (tree index:34) done accuracy:0.637489 logloss:2.82877
    [INFO random_forest.cc:578] Training of tree  41/300 (tree index:42) done accuracy:0.64191 logloss:2.37449
    [INFO random_forest.cc:578] Training of tree  51/300 (tree index:50) done accuracy:0.650752 logloss:2.05061
    [INFO random_forest.cc:578] Training of tree  61/300 (tree index:58) done accuracy:0.654288 logloss:1.72562
    [INFO random_forest.cc:578] Training of tree  71/300 (tree index:70) done accuracy:0.650752 logloss:1.6367
    [INFO random_forest.cc:578] Training of tree  81/300 (tree index:82) done accuracy:0.661362 logloss:1.62669
    [INFO random_forest.cc:578] Training of tree  91/300 (tree index:88) done accuracy:0.67374 logloss:1.56964
    [INFO random_forest.cc:578] Training of tree  101/300 (tree index:103) done accuracy:0.680813 logloss:1.45175
    [INFO random_forest.cc:578] Training of tree  111/300 (tree index:107) done accuracy:0.675508 logloss:1.42461
    [INFO random_forest.cc:578] Training of tree  121/300 (tree index:118) done accuracy:0.668435 logloss:1.42467
    [INFO random_forest.cc:578] Training of tree  131/300 (tree index:130) done accuracy:0.672856 logloss:1.36671
    [INFO random_forest.cc:578] Training of tree  141/300 (tree index:139) done accuracy:0.672856 logloss:1.30661
    [INFO random_forest.cc:578] Training of tree  151/300 (tree index:149) done accuracy:0.671088 logloss:1.27849
    [INFO random_forest.cc:578] Training of tree  161/300 (tree index:162) done accuracy:0.67374 logloss:1.27783
    [INFO random_forest.cc:578] Training of tree  171/300 (tree index:167) done accuracy:0.67374 logloss:1.27489
    [INFO random_forest.cc:578] Training of tree  182/300 (tree index:178) done accuracy:0.672856 logloss:1.21907
    [INFO random_forest.cc:578] Training of tree  192/300 (tree index:190) done accuracy:0.672856 logloss:1.19076
    [INFO random_forest.cc:578] Training of tree  202/300 (tree index:199) done accuracy:0.669319 logloss:1.19064
    [INFO random_forest.cc:578] Training of tree  212/300 (tree index:211) done accuracy:0.668435 logloss:1.13312
    [INFO random_forest.cc:578] Training of tree  222/300 (tree index:225) done accuracy:0.664898 logloss:1.10662
    [INFO random_forest.cc:578] Training of tree  232/300 (tree index:231) done accuracy:0.668435 logloss:1.10707
    [INFO random_forest.cc:578] Training of tree  242/300 (tree index:244) done accuracy:0.668435 logloss:1.10665
    [INFO random_forest.cc:578] Training of tree  252/300 (tree index:253) done accuracy:0.664898 logloss:1.10733
    [INFO random_forest.cc:578] Training of tree  262/300 (tree index:257) done accuracy:0.666667 logloss:1.10728
    [INFO random_forest.cc:578] Training of tree  272/300 (tree index:271) done accuracy:0.672856 logloss:1.10815
    [INFO random_forest.cc:578] Training of tree  282/300 (tree index:282) done accuracy:0.671972 logloss:1.10843
    [INFO random_forest.cc:578] Training of tree  292/300 (tree index:292) done accuracy:0.669319 logloss:1.10948
    [INFO random_forest.cc:578] Training of tree  300/300 (tree index:297) done accuracy:0.671972 logloss:1.10821
    [INFO random_forest.cc:645] Final OOB metrics: accuracy:0.671972 logloss:1.10821
    [INFO kernel.cc:856] Export model in log directory: /tmp/tmpzjowj8qd
    [INFO kernel.cc:864] Save model in resources
    [INFO kernel.cc:988] Loading model from path
    [INFO decision_forest.cc:590] Model loaded with 300 root(s), 80694 node(s), and 11 input feature(s).
    [INFO abstract_model.cc:993] Engine "RandomForestGeneric" built
    [INFO kernel.cc:848] Use fast generic engine
    

### Evaluation


```python
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

```

    8/8 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.6980
    
    loss: 0.0000
    accuracy: 0.6980
    


```python
tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=4)
```

    WARNING:tensorflow:6 out of the last 9 calls to <function CoreModel.yggdrasil_model_path_tensor at 0x7f55ed03c170> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:6 out of the last 9 calls to <function CoreModel.yggdrasil_model_path_tensor at 0x7f55ed03c170> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    





<script src="https://d3js.org/d3.v6.min.js"></script>
<div id="tree_plot_93a03a93fb75463faa6bc46d7d5ecbac"></div>
<script>
/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *  Plotting of decision trees generated by TF-DF.
 *
 *  A tree is a recursive structure of node objects.
 *  A node contains one or more of the following components:
 *
 *    - A value: Representing the output of the node. If the node is not a leaf,
 *      the value is only present for analysis i.e. it is not used for
 *      predictions.
 *
 *    - A condition : For non-leaf nodes, the condition (also known as split)
 *      defines a binary test to branch to the positive or negative child.
 *
 *    - An explanation: Generally a plot showing the relation between the label
 *      and the condition to give insights about the effect of the condition.
 *
 *    - Two children : For non-leaf nodes, the children nodes. The first
 *      children (i.e. "node.children[0]") is the negative children (drawn in
 *      red). The second children is the positive one (drawn in green).
 *
 */

/**
 * Plots a single decision tree into a DOM element.
 * @param {!options} options Dictionary of configurations.
 * @param {!tree} raw_tree Recursive tree structure.
 * @param {string} canvas_id Id of the output dom element.
 */
function display_tree(options, raw_tree, canvas_id) {
  console.log(options);

  // Determine the node placement.
  const tree_struct = d3.tree().nodeSize(
      [options.node_y_offset, options.node_x_offset])(d3.hierarchy(raw_tree));

  // Boundaries of the node placement.
  let x_min = Infinity;
  let x_max = -x_min;
  let y_min = Infinity;
  let y_max = -x_min;

  tree_struct.each(d => {
    if (d.x > x_max) x_max = d.x;
    if (d.x < x_min) x_min = d.x;
    if (d.y > y_max) y_max = d.y;
    if (d.y < y_min) y_min = d.y;
  });

  // Size of the plot.
  const width = y_max - y_min + options.node_x_size + options.margin * 2;
  const height = x_max - x_min + options.node_y_size + options.margin * 2 +
      options.node_y_offset - options.node_y_size;

  const plot = d3.select(canvas_id);

  // Tool tip
  options.tooltip = plot.append('div')
                        .attr('width', 100)
                        .attr('height', 100)
                        .style('padding', '4px')
                        .style('background', '#fff')
                        .style('box-shadow', '4px 4px 0px rgba(0,0,0,0.1)')
                        .style('border', '1px solid black')
                        .style('font-family', 'sans-serif')
                        .style('font-size', options.font_size)
                        .style('position', 'absolute')
                        .style('z-index', '10')
                        .attr('pointer-events', 'none')
                        .style('display', 'none');

  // Create canvas
  const svg = plot.append('svg').attr('width', width).attr('height', height);
  const graph =
      svg.style('overflow', 'visible')
          .append('g')
          .attr('font-family', 'sans-serif')
          .attr('font-size', options.font_size)
          .attr(
              'transform',
              () => `translate(${options.margin},${
                  - x_min + options.node_y_offset / 2 + options.margin})`);

  // Plot bounding box.
  if (options.show_plot_bounding_box) {
    svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'none')
        .attr('stroke-width', 1.0)
        .attr('stroke', 'black');
  }

  // Draw the edges.
  display_edges(options, graph, tree_struct);

  // Draw the nodes.
  display_nodes(options, graph, tree_struct);
}

/**
 * Draw the nodes of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_nodes(options, graph, tree_struct) {
  const nodes = graph.append('g')
                    .selectAll('g')
                    .data(tree_struct.descendants())
                    .join('g')
                    .attr('transform', d => `translate(${d.y},${d.x})`);

  nodes.append('rect')
      .attr('x', 0.5)
      .attr('y', 0.5)
      .attr('width', options.node_x_size)
      .attr('height', options.node_y_size)
      .attr('stroke', 'lightgrey')
      .attr('stroke-width', 1)
      .attr('fill', 'white')
      .attr('y', -options.node_y_size / 2);

  // Brackets on the right of condition nodes without children.
  non_leaf_node_without_children =
      nodes.filter(node => node.data.condition != null && node.children == null)
          .append('g')
          .attr('transform', `translate(${options.node_x_size},0)`);

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,10 10,10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#F00');

  non_leaf_node_without_children.append('path')
      .attr('d', 'M0,0 C 10,0 0,-10 10,-10')
      .attr('fill', 'none')
      .attr('stroke-width', 1.0)
      .attr('stroke', '#0F0');

  const node_content = nodes.append('g').attr(
      'transform',
      `translate(0,${options.node_padding - options.node_y_size / 2})`);

  node_content.append(node => create_node_element(options, node));
}

/**
 * Creates the D3 content for a single node.
 * @param {!options} options Dictionary of configurations.
 * @param {!node} node Node to draw.
 * @return {!d3} D3 content.
 */
function create_node_element(options, node) {
  // Output accumulator.
  let output = {
    // Content to draw.
    content: d3.create('svg:g'),
    // Vertical offset to the next element to draw.
    vertical_offset: 0
  };

  // Conditions.
  if (node.data.condition != null) {
    display_condition(options, node.data.condition, output);
  }

  // Values.
  if (node.data.value != null) {
    display_value(options, node.data.value, output);
  }

  // Explanations.
  if (node.data.explanation != null) {
    display_explanation(options, node.data.explanation, output);
  }

  return output.content.node();
}


/**
 * Adds a single line of text inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {!output} output Output display accumulator.
 */
function display_node_text(options, text, output) {
  output.content.append('text')
      .attr('x', options.node_padding)
      .attr('y', output.vertical_offset)
      .attr('alignment-baseline', 'hanging')
      .text(text);
  output.vertical_offset += 10;
}

/**
 * Adds a single line of text inside of a node with a tooltip.
 * @param {!options} options Dictionary of configurations.
 * @param {string} text Text to display.
 * @param {string} tooltip Text in the Tooltip.
 * @param {!output} output Output display accumulator.
 */
function display_node_text_with_tooltip(options, text, tooltip, output) {
  const item = output.content.append('text')
                   .attr('x', options.node_padding)
                   .attr('alignment-baseline', 'hanging')
                   .text(text);

  add_tooltip(options, item, () => tooltip);
  output.vertical_offset += 10;
}

/**
 * Adds a tooltip to a dom element.
 * @param {!options} options Dictionary of configurations.
 * @param {!dom} target Dom element to equip with a tooltip.
 * @param {!func} get_content Generates the html content of the tooltip.
 */
function add_tooltip(options, target, get_content) {
  function show(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.html(get_content());
  }

  function hide(d) {
    options.tooltip.style('display', 'none');
  }

  function move(d) {
    options.tooltip.style('display', 'block');
    options.tooltip.style('left', (d.pageX + 5) + 'px');
    options.tooltip.style('top', d.pageY + 'px');
  }

  target.on('mouseover', show);
  target.on('mouseout', hide);
  target.on('mousemove', move);
}

/**
 * Adds a condition inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!condition} condition Condition to display.
 * @param {!output} output Output display accumulator.
 */
function display_condition(options, condition, output) {
  threshold_format = d3.format('r');

  if (condition.type === 'IS_MISSING') {
    display_node_text(options, `${condition.attribute} is missing`, output);
    return;
  }

  if (condition.type === 'IS_TRUE') {
    display_node_text(options, `${condition.attribute} is true`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_IS_HIGHER_THAN') {
    format = d3.format('r');
    display_node_text(
        options,
        `${condition.attribute} >= ${threshold_format(condition.threshold)}`,
        output);
    return;
  }

  if (condition.type === 'CATEGORICAL_IS_IN') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} in [...]`,
        `${condition.attribute} in [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'CATEGORICAL_SET_CONTAINS') {
    display_node_text_with_tooltip(
        options, `${condition.attribute} intersect [...]`,
        `${condition.attribute} intersect [${condition.mask}]`, output);
    return;
  }

  if (condition.type === 'NUMERICAL_SPARSE_OBLIQUE') {
    display_node_text_with_tooltip(
        options, `Sparse oblique split...`,
        `[${condition.attributes}]*[${condition.weights}]>=${
            threshold_format(condition.threshold)}`,
        output);
    return;
  }

  display_node_text(
      options, `Non supported condition ${condition.type}`, output);
}

/**
 * Adds a value inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!value} value Value to display.
 * @param {!output} output Output display accumulator.
 */
function display_value(options, value, output) {
  if (value.type === 'PROBABILITY') {
    const left_margin = 0;
    const right_margin = 50;
    const plot_width = options.node_x_size - options.node_padding * 2 -
        left_margin - right_margin;

    let cusum = Array.from(d3.cumsum(value.distribution));
    cusum.unshift(0);
    const distribution_plot = output.content.append('g').attr(
        'transform', `translate(0,${output.vertical_offset + 0.5})`);

    distribution_plot.selectAll('rect')
        .data(value.distribution)
        .join('rect')
        .attr('height', 10)
        .attr(
            'x',
            (d, i) =>
                (cusum[i] * plot_width + left_margin + options.node_padding))
        .attr('width', (d, i) => d * plot_width)
        .style('fill', (d, i) => d3.schemeSet1[i]);

    const num_examples =
        output.content.append('g')
            .attr('transform', `translate(0,${output.vertical_offset})`)
            .append('text')
            .attr('x', options.node_x_size - options.node_padding)
            .attr('alignment-baseline', 'hanging')
            .attr('text-anchor', 'end')
            .text(`(${value.num_examples})`);

    const distribution_details = d3.create('ul');
    distribution_details.selectAll('li')
        .data(value.distribution)
        .join('li')
        .append('span')
        .text(
            (d, i) =>
                'class ' + i + ': ' + d3.format('.3%')(value.distribution[i]));

    add_tooltip(options, distribution_plot, () => distribution_details.html());
    add_tooltip(options, num_examples, () => 'Number of examples');

    output.vertical_offset += 10;
    return;
  }

  if (value.type === 'REGRESSION') {
    display_node_text(
        options,
        'value: ' + d3.format('r')(value.value) + ` (` +
            d3.format('.6')(value.num_examples) + `)`,
        output);
    return;
  }

  display_node_text(options, `Non supported value ${value.type}`, output);
}

/**
 * Adds an explanation inside of a node.
 * @param {!options} options Dictionary of configurations.
 * @param {!explanation} explanation Explanation to display.
 * @param {!output} output Output display accumulator.
 */
function display_explanation(options, explanation, output) {
  // Margin before the explanation.
  output.vertical_offset += 10;

  display_node_text(
      options, `Non supported explanation ${explanation.type}`, output);
}


/**
 * Draw the edges of the tree.
 * @param {!options} options Dictionary of configurations.
 * @param {!graph} graph D3 search handle containing the graph.
 * @param {!tree_struct} tree_struct Structure of the tree (node placement,
 *     data, etc.).
 */
function display_edges(options, graph, tree_struct) {
  // Draw an edge between a parent and a child node with a bezier.
  function draw_single_edge(d) {
    return 'M' + (d.source.y + options.node_x_size) + ',' + d.source.x + ' C' +
        (d.source.y + options.node_x_size + options.edge_rounding) + ',' +
        d.source.x + ' ' + (d.target.y - options.edge_rounding) + ',' +
        d.target.x + ' ' + d.target.y + ',' + d.target.x;
  }

  graph.append('g')
      .attr('fill', 'none')
      .attr('stroke-width', 1.2)
      .selectAll('path')
      .data(tree_struct.links())
      .join('path')
      .attr('d', draw_single_edge)
      .attr(
          'stroke', d => (d.target === d.source.children[0]) ? '#0F0' : '#F00');
}

display_tree({"margin": 10, "node_x_size": 160, "node_y_size": 28, "node_x_offset": 180, "node_y_offset": 33, "font_size": 10, "edge_rounding": 20, "node_padding": 2, "show_plot_bounding_box": false}, {"value": {"type": "PROBABILITY", "distribution": [0.004508566275924256, 0.027953110910730387, 0.40577096483318303, 0.4165915238954013, 0.13255184851217314, 0.012623985572587917], "num_examples": 1109.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.297559976577759}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0031746031746031746, 0.01746031746031746, 0.22698412698412698, 0.5095238095238095, 0.22063492063492063, 0.022222222222222223], "num_examples": 630.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "volatile_acidity", "threshold": 0.5349999666213989}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.009708737864077669, 0.043689320388349516, 0.38349514563106796, 0.47572815533980584, 0.07281553398058252, 0.014563106796116505], "num_examples": 206.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "total_sulfur_dioxide", "threshold": 2.802901029586792}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.011695906432748537, 0.391812865497076, 0.5146198830409356, 0.06432748538011696, 0.017543859649122806], "num_examples": 171.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.4379801750183105}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.20512820512820512, 0.5384615384615384, 0.1794871794871795, 0.07692307692307693], "num_examples": 39.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "fixed_acidity", "threshold": 6.449999809265137}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.015151515151515152, 0.44696969696969696, 0.5075757575757576, 0.030303030303030304, 0.0], "num_examples": 132.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.3174614906311035}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.05714285714285714, 0.2, 0.34285714285714286, 0.2857142857142857, 0.11428571428571428, 0.0], "num_examples": 35.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "fixed_acidity", "threshold": 7.350000381469727}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.14285714285714285, 0.0, 0.35714285714285715, 0.42857142857142855, 0.07142857142857142, 0.0], "num_examples": 14.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "volatile_acidity", "threshold": 0.6499999761581421}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.3333333333333333, 0.3333333333333333, 0.19047619047619047, 0.14285714285714285, 0.0], "num_examples": 21.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "density", "threshold": 0.9953550100326538}}]}]}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0047169811320754715, 0.1509433962264151, 0.5259433962264151, 0.29245283018867924, 0.025943396226415096], "num_examples": 424.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.420358180999756}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.011049723756906077, 0.027624309392265192, 0.5248618784530387, 0.39779005524861877, 0.03867403314917127], "num_examples": 181.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "sulphates", "threshold": 0.6150000095367432}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.016, 0.448, 0.48, 0.056], "num_examples": 125.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "total_sulfur_dioxide", "threshold": 4.317399024963379}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.03571428571428571, 0.05357142857142857, 0.6964285714285714, 0.21428571428571427, 0.0], "num_examples": 56.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "pH", "threshold": 3.2649998664855957}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.24279835390946503, 0.5267489711934157, 0.2139917695473251, 0.01646090534979424], "num_examples": 243.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "volatile_acidity", "threshold": 0.36250001192092896}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.26143790849673204, 0.6013071895424836, 0.13725490196078433, 0.0], "num_examples": 153.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.3042490482330322}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.2111111111111111, 0.4, 0.34444444444444444, 0.044444444444444446], "num_examples": 90.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "residual_sugar", "threshold": 3.0999999046325684}}]}]}]}, {"value": {"type": "PROBABILITY", "distribution": [0.006263048016701462, 0.04175365344467641, 0.6409185803757829, 0.29436325678496866, 0.016701461377870562, 0.0], "num_examples": 479.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "sulphates", "threshold": 0.5249999761581421}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0027100271002710027, 0.02168021680216802, 0.6097560975609756, 0.34417344173441733, 0.02168021680216802, 0.0], "num_examples": 369.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "fixed_acidity", "threshold": 10.850000381469727}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.02857142857142857, 0.0, 0.08571428571428572, 0.7142857142857143, 0.17142857142857143, 0.0], "num_examples": 35.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "sulphates", "threshold": 0.7549999952316284}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.0, 0.0, 0.45454545454545453, 0.5454545454545454, 0.0], "num_examples": 11.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "density", "threshold": 1.0004249811172485}}, {"value": {"type": "PROBABILITY", "distribution": [0.041666666666666664, 0.0, 0.125, 0.8333333333333334, 0.0, 0.0], "num_examples": 24.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "total_sulfur_dioxide", "threshold": 3.7373862266540527}}]}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.023952095808383235, 0.6646706586826348, 0.30538922155688625, 0.005988023952095809, 0.0], "num_examples": 334.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "citric_acid", "threshold": 0.48500001430511475}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.07547169811320754, 0.8301886792452831, 0.07547169811320754, 0.018867924528301886, 0.0], "num_examples": 53.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "fixed_acidity", "threshold": 9.149999618530273}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.014234875444839857, 0.6334519572953736, 0.3487544483985765, 0.0035587188612099642, 0.0], "num_examples": 281.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "density", "threshold": 0.9966049790382385}}]}]}, {"value": {"type": "PROBABILITY", "distribution": [0.01818181818181818, 0.10909090909090909, 0.7454545454545455, 0.12727272727272726, 0.0, 0.0], "num_examples": 110.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "volatile_acidity", "threshold": 0.8774999976158142}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.0, 0.8571428571428571, 0.14285714285714285, 0.0, 0.0, 0.0], "num_examples": 7.0}}, {"value": {"type": "PROBABILITY", "distribution": [0.019417475728155338, 0.05825242718446602, 0.7864077669902912, 0.13592233009708737, 0.0, 0.0], "num_examples": 103.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "alcohol", "threshold": 2.266944408416748}, "children": [{"value": {"type": "PROBABILITY", "distribution": [0.09523809523809523, 0.047619047619047616, 0.47619047619047616, 0.38095238095238093, 0.0, 0.0], "num_examples": 21.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "pH", "threshold": 3.1449999809265137}}, {"value": {"type": "PROBABILITY", "distribution": [0.0, 0.06097560975609756, 0.8658536585365854, 0.07317073170731707, 0.0, 0.0], "num_examples": 82.0}, "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "residual_sugar", "threshold": 2.450000047683716}}]}]}]}]}, "#tree_plot_93a03a93fb75463faa6bc46d7d5ecbac")
</script>




### Prediction measurements


```python
predict_prob = model_1.predict(x = test_ds)

# Get model output for test sample
vec_pred_class_idx, vec_pred_class, vec_pred_unc = vec_prediction(predict_prob)

# Reset index for use:
test_ds_red.reset_index(drop = True, inplace = True)
# save the result in the dataframe
test_ds_red['quality_pred'] = vec_pred_class_idx
test_ds_red['pred_class_correct'] = [1 if (val == test_ds_red.quality[iter]) 
                                        else 0 for (iter, val) in enumerate(test_ds_red.quality_pred)]

print(f'Test set accuracy: {round( np.mean(test_ds_red.pred_class_correct) *100, 2)} percent')

# 67.9 percent for no data transformation (log-scale)
```


```python
model_1.summary()
```

    Model: "random_forest_model_8"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Total params: 1
    Trainable params: 0
    Non-trainable params: 1
    _________________________________________________________________
    Type: "RANDOM_FOREST"
    Task: CLASSIFICATION
    Label: "__LABEL"
    
    Input Features (11):
    	alcohol
    	chlorides
    	citric_acid
    	density
    	fixed_acidity
    	free_sulfur_dioxide
    	pH
    	residual_sugar
    	sulphates
    	total_sulfur_dioxide
    	volatile_acidity
    
    No weights
    
    Variable Importance: MEAN_MIN_DEPTH:
        1.              "__LABEL"  8.652279 ################
        2.                   "pH"  7.360571 #############
        3.  "free_sulfur_dioxide"  7.262763 ############
        4.       "residual_sugar"  6.991891 ############
        5.            "chlorides"  6.745837 ###########
        6.        "fixed_acidity"  6.635087 ###########
        7.          "citric_acid"  6.443833 ##########
        8.              "density"  5.507326 ########
        9. "total_sulfur_dioxide"  5.117035 #######
       10.            "sulphates"  3.129226 ###
       11.     "volatile_acidity"  3.023740 ##
       12.              "alcohol"  1.742996 
    
    Variable Importance: NUM_AS_ROOT:
        1.              "alcohol" 127.000000 ################
        2.     "volatile_acidity" 70.000000 ########
        3.            "sulphates" 60.000000 #######
        4.          "citric_acid" 17.000000 #
        5.              "density" 16.000000 #
        6. "total_sulfur_dioxide"  7.000000 
        7.            "chlorides"  3.000000 
    
    Variable Importance: NUM_NODES:
        1.     "volatile_acidity" 4281.000000 ################
        2. "total_sulfur_dioxide" 4123.000000 #############
        3.            "sulphates" 4088.000000 #############
        4.              "density" 3874.000000 ##########
        5.              "alcohol" 3833.000000 ##########
        6.            "chlorides" 3629.000000 #######
        7.        "fixed_acidity" 3519.000000 ######
        8.          "citric_acid" 3339.000000 ###
        9.                   "pH" 3295.000000 ###
       10.       "residual_sugar" 3068.000000 
       11.  "free_sulfur_dioxide" 3042.000000 
    
    Variable Importance: SUM_SCORE:
        1.              "alcohol" 53046.472666 ################
        2.     "volatile_acidity" 38869.181521 #########
        3.            "sulphates" 36047.602842 ########
        4. "total_sulfur_dioxide" 28061.052167 ####
        5.              "density" 25359.759206 ###
        6.            "chlorides" 20924.211244 #
        7.        "fixed_acidity" 20625.991136 #
        8.          "citric_acid" 19717.884387 #
        9.                   "pH" 18045.344245 
       10.       "residual_sugar" 17859.605762 
       11.  "free_sulfur_dioxide" 17321.720276 
    
    
    
    Winner take all: true
    Out-of-bag evaluation: accuracy:0.646528 logloss:1.2847
    Number of trees: 300
    Total number of nodes: 80482
    
    Number of nodes by tree:
    Count: 300 Average: 268.273 StdDev: 9.46248
    Min: 237 Max: 291 Ignored: 0
    ----------------------------------------------
    [ 237, 239)  1   0.33%   0.33%
    [ 239, 242)  0   0.00%   0.33%
    [ 242, 245)  2   0.67%   1.00%
    [ 245, 248)  4   1.33%   2.33% #
    [ 248, 250)  3   1.00%   3.33% #
    [ 250, 253)  7   2.33%   5.67% #
    [ 253, 256) 12   4.00%   9.67% ##
    [ 256, 259) 13   4.33%  14.00% ###
    [ 259, 261) 15   5.00%  19.00% ###
    [ 261, 264) 34  11.33%  30.33% #######
    [ 264, 267) 28   9.33%  39.67% ######
    [ 267, 270) 49  16.33%  56.00% ##########
    [ 270, 272) 23   7.67%  63.67% #####
    [ 272, 275) 19   6.33%  70.00% ####
    [ 275, 278) 45  15.00%  85.00% #########
    [ 278, 281) 14   4.67%  89.67% ###
    [ 281, 283) 14   4.67%  94.33% ###
    [ 283, 286) 10   3.33%  97.67% ##
    [ 286, 289)  4   1.33%  99.00% #
    [ 289, 291]  3   1.00% 100.00% #
    
    Depth by leafs:
    Count: 40391 Average: 8.65297 StdDev: 2.37179
    Min: 2 Max: 15 Ignored: 0
    ----------------------------------------------
    [  2,  3)    8   0.02%   0.02%
    [  3,  4)  148   0.37%   0.39%
    [  4,  5)  770   1.91%   2.29% #
    [  5,  6) 2225   5.51%   7.80% ###
    [  6,  7) 4301  10.65%  18.45% ######
    [  7,  8) 6156  15.24%  33.69% #########
    [  8,  9) 6819  16.88%  50.57% ##########
    [  9, 10) 6342  15.70%  66.27% #########
    [ 10, 11) 5184  12.83%  79.11% ########
    [ 11, 12) 3410   8.44%  87.55% #####
    [ 12, 13) 2293   5.68%  93.23% ###
    [ 13, 14) 1419   3.51%  96.74% ##
    [ 14, 15)  744   1.84%  98.58% #
    [ 15, 15]  572   1.42% 100.00% #
    
    Number of training obs by leaf:
    Count: 40391 Average: 8.23698 StdDev: 5.45889
    Min: 5 Max: 83 Ignored: 0
    ----------------------------------------------
    [  5,  8) 25502  63.14%  63.14% ##########
    [  8, 12)  9592  23.75%  86.89% ####
    [ 12, 16)  2217   5.49%  92.37% #
    [ 16, 20)  1262   3.12%  95.50%
    [ 20, 24)   699   1.73%  97.23%
    [ 24, 28)   401   0.99%  98.22%
    [ 28, 32)   270   0.67%  98.89%
    [ 32, 36)   155   0.38%  99.27%
    [ 36, 40)   106   0.26%  99.54%
    [ 40, 44)    67   0.17%  99.70%
    [ 44, 48)    52   0.13%  99.83%
    [ 48, 52)    28   0.07%  99.90%
    [ 52, 56)    20   0.05%  99.95%
    [ 56, 60)     5   0.01%  99.96%
    [ 60, 64)     3   0.01%  99.97%
    [ 64, 68)     9   0.02%  99.99%
    [ 68, 72)     1   0.00% 100.00%
    [ 72, 76)     0   0.00% 100.00%
    [ 76, 80)     0   0.00% 100.00%
    [ 80, 83]     2   0.00% 100.00%
    
    Attribute in nodes:
    	4281 : volatile_acidity [NUMERICAL]
    	4123 : total_sulfur_dioxide [NUMERICAL]
    	4088 : sulphates [NUMERICAL]
    	3874 : density [NUMERICAL]
    	3833 : alcohol [NUMERICAL]
    	3629 : chlorides [NUMERICAL]
    	3519 : fixed_acidity [NUMERICAL]
    	3339 : citric_acid [NUMERICAL]
    	3295 : pH [NUMERICAL]
    	3068 : residual_sugar [NUMERICAL]
    	3042 : free_sulfur_dioxide [NUMERICAL]
    
    Attribute in nodes with depth <= 0:
    	127 : alcohol [NUMERICAL]
    	70 : volatile_acidity [NUMERICAL]
    	60 : sulphates [NUMERICAL]
    	17 : citric_acid [NUMERICAL]
    	16 : density [NUMERICAL]
    	7 : total_sulfur_dioxide [NUMERICAL]
    	3 : chlorides [NUMERICAL]
    
    Attribute in nodes with depth <= 1:
    	286 : alcohol [NUMERICAL]
    	190 : sulphates [NUMERICAL]
    	189 : volatile_acidity [NUMERICAL]
    	64 : density [NUMERICAL]
    	61 : total_sulfur_dioxide [NUMERICAL]
    	49 : citric_acid [NUMERICAL]
    	26 : fixed_acidity [NUMERICAL]
    	21 : chlorides [NUMERICAL]
    	6 : residual_sugar [NUMERICAL]
    	6 : free_sulfur_dioxide [NUMERICAL]
    	2 : pH [NUMERICAL]
    
    Attribute in nodes with depth <= 2:
    	532 : alcohol [NUMERICAL]
    	397 : volatile_acidity [NUMERICAL]
    	348 : sulphates [NUMERICAL]
    	201 : density [NUMERICAL]
    	184 : total_sulfur_dioxide [NUMERICAL]
    	99 : fixed_acidity [NUMERICAL]
    	90 : citric_acid [NUMERICAL]
    	76 : chlorides [NUMERICAL]
    	66 : free_sulfur_dioxide [NUMERICAL]
    	62 : residual_sugar [NUMERICAL]
    	37 : pH [NUMERICAL]
    
    Attribute in nodes with depth <= 3:
    	837 : alcohol [NUMERICAL]
    	711 : volatile_acidity [NUMERICAL]
    	619 : sulphates [NUMERICAL]
    	447 : total_sulfur_dioxide [NUMERICAL]
    	406 : density [NUMERICAL]
    	264 : fixed_acidity [NUMERICAL]
    	234 : chlorides [NUMERICAL]
    	232 : citric_acid [NUMERICAL]
    	227 : residual_sugar [NUMERICAL]
    	185 : free_sulfur_dioxide [NUMERICAL]
    	166 : pH [NUMERICAL]
    
    Attribute in nodes with depth <= 5:
    	1714 : volatile_acidity [NUMERICAL]
    	1706 : alcohol [NUMERICAL]
    	1545 : sulphates [NUMERICAL]
    	1470 : total_sulfur_dioxide [NUMERICAL]
    	1231 : density [NUMERICAL]
    	1021 : fixed_acidity [NUMERICAL]
    	1002 : chlorides [NUMERICAL]
    	952 : residual_sugar [NUMERICAL]
    	871 : pH [NUMERICAL]
    	864 : citric_acid [NUMERICAL]
    	833 : free_sulfur_dioxide [NUMERICAL]
    
    Condition type in nodes:
    	40091 : HigherCondition
    Condition type in nodes with depth <= 0:
    	300 : HigherCondition
    Condition type in nodes with depth <= 1:
    	900 : HigherCondition
    Condition type in nodes with depth <= 2:
    	2092 : HigherCondition
    Condition type in nodes with depth <= 3:
    	4328 : HigherCondition
    Condition type in nodes with depth <= 5:
    	13209 : HigherCondition
    Node format: NOT_SET
    
    Training OOB:
    	trees: 1, Out-of-bag evaluation: accuracy:0.544335 logloss:16.4238
    	trees: 11, Out-of-bag evaluation: accuracy:0.573115 logloss:7.16348
    	trees: 21, Out-of-bag evaluation: accuracy:0.601443 logloss:3.62689
    	trees: 31, Out-of-bag evaluation: accuracy:0.616772 logloss:2.71117
    	trees: 41, Out-of-bag evaluation: accuracy:0.635708 logloss:2.22787
    	trees: 51, Out-of-bag evaluation: accuracy:0.633904 logloss:1.89418
    	trees: 61, Out-of-bag evaluation: accuracy:0.64743 logloss:1.82451
    	trees: 71, Out-of-bag evaluation: accuracy:0.642922 logloss:1.76484
    	trees: 81, Out-of-bag evaluation: accuracy:0.629396 logloss:1.76258
    	trees: 91, Out-of-bag evaluation: accuracy:0.64202 logloss:1.70319
    	trees: 101, Out-of-bag evaluation: accuracy:0.63661 logloss:1.70255
    	trees: 111, Out-of-bag evaluation: accuracy:0.63661 logloss:1.66784
    	trees: 121, Out-of-bag evaluation: accuracy:0.63661 logloss:1.60689
    	trees: 131, Out-of-bag evaluation: accuracy:0.642922 logloss:1.60264
    	trees: 141, Out-of-bag evaluation: accuracy:0.642922 logloss:1.60036
    	trees: 151, Out-of-bag evaluation: accuracy:0.642922 logloss:1.59919
    	trees: 161, Out-of-bag evaluation: accuracy:0.648332 logloss:1.57038
    	trees: 171, Out-of-bag evaluation: accuracy:0.64743 logloss:1.51168
    	trees: 181, Out-of-bag evaluation: accuracy:0.64743 logloss:1.45661
    	trees: 191, Out-of-bag evaluation: accuracy:0.646528 logloss:1.45571
    	trees: 201, Out-of-bag evaluation: accuracy:0.651939 logloss:1.39766
    	trees: 211, Out-of-bag evaluation: accuracy:0.648332 logloss:1.34123
    	trees: 221, Out-of-bag evaluation: accuracy:0.646528 logloss:1.31393
    	trees: 231, Out-of-bag evaluation: accuracy:0.651939 logloss:1.31311
    	trees: 241, Out-of-bag evaluation: accuracy:0.64743 logloss:1.31212
    	trees: 251, Out-of-bag evaluation: accuracy:0.64743 logloss:1.28334
    	trees: 261, Out-of-bag evaluation: accuracy:0.650135 logloss:1.28522
    	trees: 271, Out-of-bag evaluation: accuracy:0.648332 logloss:1.28616
    	trees: 281, Out-of-bag evaluation: accuracy:0.651037 logloss:1.28537
    	trees: 291, Out-of-bag evaluation: accuracy:0.648332 logloss:1.28565
    	trees: 300, Out-of-bag evaluation: accuracy:0.646528 logloss:1.2847
    
    


```python
# The input features
model_1.make_inspector().features()
```




    ["alcohol" (1; #0),
     "chlorides" (1; #1),
     "citric_acid" (1; #2),
     "density" (1; #3),
     "fixed_acidity" (1; #4),
     "free_sulfur_dioxide" (1; #5),
     "pH" (1; #6),
     "residual_sugar" (1; #7),
     "sulphates" (1; #8),
     "total_sulfur_dioxide" (1; #9),
     "volatile_acidity" (1; #10)]




```python
# The feature importances
model_1.make_inspector().variable_importances()
```




    {'MEAN_MIN_DEPTH': [("__LABEL" (4; #11), 8.652278543134686),
      ("pH" (1; #6), 7.3605714742744635),
      ("free_sulfur_dioxide" (1; #5), 7.262762710809184),
      ("residual_sugar" (1; #7), 6.9918910597238),
      ("chlorides" (1; #1), 6.745837306352339),
      ("fixed_acidity" (1; #4), 6.635086705089281),
      ("citric_acid" (1; #2), 6.4438334742618455),
      ("density" (1; #3), 5.507325545852401),
      ("total_sulfur_dioxide" (1; #9), 5.117035088168903),
      ("sulphates" (1; #8), 3.1292264765776863),
      ("volatile_acidity" (1; #10), 3.0237400253144253),
      ("alcohol" (1; #0), 1.7429956813259047)],
     'NUM_AS_ROOT': [("alcohol" (1; #0), 127.0),
      ("volatile_acidity" (1; #10), 70.0),
      ("sulphates" (1; #8), 60.0),
      ("citric_acid" (1; #2), 17.0),
      ("density" (1; #3), 16.0),
      ("total_sulfur_dioxide" (1; #9), 7.0),
      ("chlorides" (1; #1), 3.0)],
     'NUM_NODES': [("volatile_acidity" (1; #10), 4281.0),
      ("total_sulfur_dioxide" (1; #9), 4123.0),
      ("sulphates" (1; #8), 4088.0),
      ("density" (1; #3), 3874.0),
      ("alcohol" (1; #0), 3833.0),
      ("chlorides" (1; #1), 3629.0),
      ("fixed_acidity" (1; #4), 3519.0),
      ("citric_acid" (1; #2), 3339.0),
      ("pH" (1; #6), 3295.0),
      ("residual_sugar" (1; #7), 3068.0),
      ("free_sulfur_dioxide" (1; #5), 3042.0)],
     'SUM_SCORE': [("alcohol" (1; #0), 53046.47266591876),
      ("volatile_acidity" (1; #10), 38869.18152080104),
      ("sulphates" (1; #8), 36047.602842281514),
      ("total_sulfur_dioxide" (1; #9), 28061.052167497284),
      ("density" (1; #3), 25359.759205840528),
      ("chlorides" (1; #1), 20924.211244474107),
      ("fixed_acidity" (1; #4), 20625.991135557648),
      ("citric_acid" (1; #2), 19717.88438662974),
      ("pH" (1; #6), 18045.344245144806),
      ("residual_sugar" (1; #7), 17859.60576188448),
      ("free_sulfur_dioxide" (1; #5), 17321.720275617205)]}



### Training and validation data


```python
# This cell start TensorBoard that can be slow.
# Load the TensorBoard notebook extension
%load_ext tensorboard
# Google internal version
# %load_ext google3.learning.brain.tensorboard.notebook.extension

# Clear existing results (if any)
rm -fr "/tmp/tensorboard_logs"

# Export the meta-data to tensorboard.
model_1.make_inspector().export_to_tensorboard("/tmp/tensorboard_logs")

# Start a tensorboard instance.
%tensorboard --logdir "/tmp/tensorboard_logs"
```


      File "<ipython-input-65-887c4c481378>", line 8
        rm -fr "/tmp/tensorboard_logs"
                                     ^
    SyntaxError: invalid syntax
    


## Iteration 2: combining Red wine & White wine data 



```python
df_red = pd.read_csv(str_df_red, sep = ';')
df_wht = pd.read_csv(str_df_white, sep = ';')

# Create indicator for each
df_red['idx_red'] = 1
df_wht['idx_red'] = 0
df = pd.concat([df_red, df_wht])

df['alcohol'] = np.log(df['alcohol'])
df['free sulfur dioxide'] = np.log(df['free sulfur dioxide'])
df['total sulfur dioxide'] = np.log(df['total sulfur dioxide'] )

# White wine
df.loc[:,'quality'] = df['quality'].map(str)
classes = sorted(df['quality'].unique().tolist())
print(classes)

df['quality'] = df.quality.map(classes.index)
```

    ['3', '4', '5', '6', '7', '8', '9']
    

### Model fit


```python
def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_df, test_ds_df = split_dataset(df)

# Variable of interest 
label = 'quality'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_df, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_df, label=label)

# Specify the model.
model_2 = tfdf.keras.RandomForestModel()

# Optionally, add evaluation metrics.
model_2.compile(metrics=["accuracy"])

# Train the model.
# "sys_pipes" is optional. It enables the display of the training logs.
with sys_pipes():
  model_2.fit(x=train_ds)

# Evaluate the model
evaluation = model_2.evaluate(test_ds, return_dict=True)
```

    WARNING:absl:Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    WARNING:absl:Some of the feature names have been changed automatically to be compatible with SavedModels because fix_feature_names=True.
    2021-09-25 02:37:06.398187: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
    

    72/72 [==============================] - 5s 2ms/step
    

    [INFO kernel.cc:746] Start Yggdrasil model training
    [INFO kernel.cc:747] Collect training examples
    [INFO kernel.cc:392] Number of batches: 72
    [INFO kernel.cc:393] Number of examples: 4580
    [INFO kernel.cc:769] Dataset:
    Number of records: 4580
    Number of columns: 13
    
    Number of columns by type:
    	NUMERICAL: 12 (92.3077%)
    	CATEGORICAL: 1 (7.69231%)
    
    Columns:
    
    NUMERICAL: 12 (92.3077%)
    	0: "alcohol" NUMERICAL mean:2.34428 min:2.07944 max:2.65324 sd:0.110782
    	1: "chlorides" NUMERICAL mean:0.0564908 min:0.009 max:0.611 sd:0.0371518
    	2: "citric_acid" NUMERICAL mean:0.318777 min:0 max:1.66 sd:0.145913
    	3: "density" NUMERICAL mean:0.994703 min:0.98711 max:1.03898 sd:0.00301115
    	4: "fixed_acidity" NUMERICAL mean:7.21879 min:3.8 max:15.6 sd:1.28733
    	5: "free_sulfur_dioxide" NUMERICAL mean:3.20882 min:0 max:4.98703 sd:0.699386
    	6: "idx_red" NUMERICAL mean:0.247162 min:0 max:1 sd:0.431361
    	7: "pH" NUMERICAL mean:3.21849 min:2.72 max:4.01 sd:0.162256
    	8: "residual_sugar" NUMERICAL mean:5.44361 min:0.6 max:65.8 sd:4.77453
    	9: "sulphates" NUMERICAL mean:0.531773 min:0.22 max:2 sd:0.149394
    	10: "total_sulfur_dioxide" NUMERICAL mean:4.55945 min:1.79176 max:5.84064 sd:0.713849
    	11: "volatile_acidity" NUMERICAL mean:0.339183 min:0.08 max:1.58 sd:0.163648
    
    CATEGORICAL: 1 (7.69231%)
    	12: "__LABEL" CATEGORICAL integerized vocab-size:8 no-ood-item
    
    Terminology:
    	nas: Number of non-available (i.e. missing) values.
    	ood: Out of dictionary.
    	manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    	tokenized: The attribute value is obtained through tokenization.
    	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    	vocab-size: Number of unique values.
    
    [INFO kernel.cc:772] Configure learner
    [INFO kernel.cc:797] Training config:
    learner: "RANDOM_FOREST"
    features: "alcohol"
    features: "chlorides"
    features: "citric_acid"
    features: "density"
    features: "fixed_acidity"
    features: "free_sulfur_dioxide"
    features: "idx_red"
    features: "pH"
    features: "residual_sugar"
    features: "sulphates"
    features: "total_sulfur_dioxide"
    features: "volatile_acidity"
    label: "__LABEL"
    task: CLASSIFICATION
    [yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
      num_trees: 300
      decision_tree {
        max_depth: 16
        min_examples: 5
        in_split_min_examples_check: true
        missing_value_policy: GLOBAL_IMPUTATION
        allow_na_conditions: false
        categorical_set_greedy_forward {
          sampling: 0.1
          max_num_items: -1
          min_item_frequency: 1
        }
        growing_strategy_local {
        }
        categorical {
          cart {
          }
        }
        num_candidate_attributes_ratio: -1
        axis_aligned_split {
        }
        internal {
          sorting_strategy: PRESORTED
        }
      }
      winner_take_all_inference: true
      compute_oob_performances: true
      compute_oob_variable_importances: false
      adapt_bootstrap_size_ratio_for_maximum_training_duration: false
    }
    
    [INFO kernel.cc:800] Deployment config:
    num_threads: 6
    
    [INFO kernel.cc:837] Train model
    [INFO random_forest.cc:303] Training random forest on 4580 example(s) and 12 feature(s).
    [INFO random_forest.cc:578] Training of tree  1/300 (tree index:0) done accuracy:0.492289 logloss:18.2997
    [INFO random_forest.cc:578] Training of tree  11/300 (tree index:11) done accuracy:0.569197 logloss:6.48371
    [INFO random_forest.cc:578] Training of tree  21/300 (tree index:20) done accuracy:0.609825 logloss:3.62394
    [INFO random_forest.cc:578] Training of tree  31/300 (tree index:30) done accuracy:0.620961 logloss:2.81572
    [INFO random_forest.cc:578] Training of tree  41/300 (tree index:39) done accuracy:0.634934 logloss:2.31068
    [INFO random_forest.cc:578] Training of tree  51/300 (tree index:50) done accuracy:0.64083 logloss:2.06653
    [INFO random_forest.cc:578] Training of tree  61/300 (tree index:61) done accuracy:0.645633 logloss:1.89886
    [INFO random_forest.cc:578] Training of tree  71/300 (tree index:70) done accuracy:0.648035 logloss:1.75939
    [INFO random_forest.cc:578] Training of tree  81/300 (tree index:80) done accuracy:0.64869 logloss:1.6375
    [INFO random_forest.cc:578] Training of tree  91/300 (tree index:91) done accuracy:0.649345 logloss:1.51461
    [INFO random_forest.cc:578] Training of tree  101/300 (tree index:101) done accuracy:0.652402 logloss:1.44996
    [INFO random_forest.cc:578] Training of tree  111/300 (tree index:110) done accuracy:0.65393 logloss:1.42782
    [INFO random_forest.cc:578] Training of tree  121/300 (tree index:120) done accuracy:0.650437 logloss:1.41287
    [INFO random_forest.cc:578] Training of tree  131/300 (tree index:133) done accuracy:0.652402 logloss:1.37089
    [INFO random_forest.cc:578] Training of tree  141/300 (tree index:137) done accuracy:0.649563 logloss:1.36287
    [INFO random_forest.cc:578] Training of tree  151/300 (tree index:149) done accuracy:0.648253 logloss:1.31284
    [INFO random_forest.cc:578] Training of tree  161/300 (tree index:160) done accuracy:0.648908 logloss:1.31146
    [INFO random_forest.cc:578] Training of tree  171/300 (tree index:170) done accuracy:0.651747 logloss:1.28163
    [INFO random_forest.cc:578] Training of tree  181/300 (tree index:180) done accuracy:0.653712 logloss:1.25265
    [INFO random_forest.cc:578] Training of tree  191/300 (tree index:190) done accuracy:0.653275 logloss:1.25218
    [INFO random_forest.cc:578] Training of tree  201/300 (tree index:199) done accuracy:0.651747 logloss:1.23155
    [INFO random_forest.cc:578] Training of tree  211/300 (tree index:210) done accuracy:0.653712 logloss:1.21708
    [INFO random_forest.cc:578] Training of tree  221/300 (tree index:220) done accuracy:0.65 logloss:1.20342
    [INFO random_forest.cc:578] Training of tree  231/300 (tree index:230) done accuracy:0.65131 logloss:1.20331
    [INFO random_forest.cc:578] Training of tree  241/300 (tree index:240) done accuracy:0.651528 logloss:1.19558
    [INFO random_forest.cc:578] Training of tree  251/300 (tree index:250) done accuracy:0.652402 logloss:1.1755
    [INFO random_forest.cc:578] Training of tree  261/300 (tree index:260) done accuracy:0.652402 logloss:1.16903
    [INFO random_forest.cc:578] Training of tree  271/300 (tree index:270) done accuracy:0.653057 logloss:1.16854
    [INFO random_forest.cc:578] Training of tree  281/300 (tree index:280) done accuracy:0.653057 logloss:1.16904
    [INFO random_forest.cc:578] Training of tree  291/300 (tree index:290) done accuracy:0.653275 logloss:1.15508
    [INFO random_forest.cc:578] Training of tree  300/300 (tree index:298) done accuracy:0.652838 logloss:1.14064
    [INFO random_forest.cc:645] Final OOB metrics: accuracy:0.652838 logloss:1.14064
    [INFO kernel.cc:856] Export model in log directory: /tmp/tmpv9xhng7q
    [INFO kernel.cc:864] Save model in resources
    [INFO kernel.cc:988] Loading model from path
    [INFO decision_forest.cc:590] Model loaded with 300 root(s), 322954 node(s), and 12 input feature(s).
    [INFO abstract_model.cc:993] Engine "RandomForestGeneric" built
    [INFO kernel.cc:848] Use fast generic engine
    

    30/30 [==============================] - 1s 9ms/step - loss: 0.0000e+00 - accuracy: 0.6479
    


```python
train_ds_eval = train_ds_df.copy()

predict_prob = model_2.predict(x = train_ds)

# Get model output for test sample
vec_pred_class_idx, vec_pred_class, vec_pred_unc = vec_prediction(predict_prob)

# Reset index for use:
train_ds_eval.reset_index(drop = True, inplace = True)
# save the result in the dataframe
train_ds_eval['quality_pred'] = vec_pred_class_idx
train_ds_eval['pred_class_correct'] = [1 if (val == train_ds_eval.quality[iter]) 
                                        else 0 for (iter, val) in enumerate(train_ds_eval.quality_pred)]

print(f'Train set accuracy: {round( np.mean(train_ds_eval.pred_class_correct) *100, 2)} percent')

```

    Test set accuracy: 91.72 percent
    


```python
test_ds_eval = test_ds_df.copy()

predict_prob = model_2.predict(x = test_ds)

# Get model output for test sample
vec_pred_class_idx, vec_pred_class, vec_pred_unc = vec_prediction(predict_prob)

# Reset index for use:
test_ds_eval.reset_index(drop = True, inplace = True)
# save the result in the dataframe
test_ds_eval['quality_pred'] = vec_pred_class_idx
test_ds_eval['pred_class_correct'] = [1 if (val == test_ds_eval.quality[iter]) 
                                        else 0 for (iter, val) in enumerate(test_ds_eval.quality_pred)]

print(f'Test set accuracy: {round( np.mean(test_ds_eval.pred_class_correct) *100, 2)} percent')

# 67.9 percent for no data transformation (log-scale)
```

    Test set accuracy: 64.79 percent
    
