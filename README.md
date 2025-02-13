# MC-GMENN: Enabling Mixed Effects Neural Networks for Diverse, Clustered Data Using Monte Carlo Methods

This repository contains the code to reproduce the experiments in our IJCAI 2024 paper. 

To replicate the experiments reported in the paper and in the supplementary material:
1. Create a Python 3.11 environment and install the requirements: 'pip install -r requirements.txt'
2. Need to run the following on macOS: 
    - `brew install cmake` (to suppress an error emanating from `python setup.py bdist_wheel` in `dm-tree`); 
    - `brew install libomp` (to remedy `OSError` emanating from `gpboost`).This error will surface much later, while running the first cell in `example.ipynb`!
3. Run "python download_pargent2022_datasets.py" to download the datasets used in the high-cardinality categorical features benchmark study.
4. Afterwards, the notebooks can be executed to reproduce the results. Note that the GPU configurations in the first cell of each notebook might need to be adjusted for the user-specific setup. 

In the future, the MC-GMENN model will also be made available as a pip package. In the meantime, 'notebooks/example.py' contains a simplified example on how to apply our model to a dataset.

The repository also contains a copy of LMMNN (https://github.com/gsimchoni/lmmnn) with the following changes made:
- function reg_nn_lmmnn in nn.py changed to return b_hat and the model itself
- adapt binary data generation function to return b_hat
- function reg_nn_lmmnn in nn.py changed to include option to pass a base model, optimizer and validation data as a parameter to allow easier comparison to other methods 

Furthermore, a copy of ARMED (https://gitfront.io/r/DeepLearningForPrecisionHealthLab/54f18307815dfb2148fbc2d14368c1268b63825e/ARMED-MixedEffectsDL/) is included, where we made additions be able to run random intercept models with the base neural networks described in our paper.

# Usage: _Your_ $X$, $Z$ and $Y$

The `process_dataset` function in `data/preprocessing/dataset_preprocessing.py` is the entry point. Be warned, the function is ~1K lines long!

Let's read this funtion line by line:
1. Lines 1-63: Path for the processed data is set
2. Line 78: Column named `Unnamed: 0` is dropped
3. Line 85: Columns with > 5% missing values are dropped
4. Lines 87-655: Bulk of the code is here, and it's a giant ladder of `if` statements that process one dataset each! _Add your data processing logic here._ An example can be found [here](#example-processing-road-safety-drivers-sex). Note that columns are divided into 5 categories:
    - `y_col`: The target variable. Must be encoded with `LabelEncoder`, if a string.
    - `bin_cols`: Variables with two unique values – i.e., _binary_.
    - `z_cols`: Columns with high-dimensional categorical value – i.e., more than `hct` unique values.
    - `cat_cols`: Columns with low-dimensional categorical variables.
    - `numeric_cols`: Numerical variables.
5. Lines 656-694: The data is split into train, validation, and test sets – randomly with `RS` as the seed. _Change the code to accept splits from you._
6. Lines 695-857: Another dizzying lump of code that handles enncoding, imputation and standardization. Interesting fact: categorical variables are first encoded with `OrdinalEncoder`!
7. Lines 858-890: Columns with constant values are dropped.
8. Lines 894-921: $X$ is finally defined. Consists of everything other than the high-dmensional categorical variables.
9. Lines 922-932: $Z$ is defined. Consists of the high-dimensional categorical variables.
10. Rest of the lines: Another dizzying lump. By this time, you're probably numb.

Sample log from steps 8-9:-
```bash
2025-02-13 13:21:51.848 | INFO     | data.preprocessing.dataset_preprocessing:process_dataset:934 - Z variables: ['number_customer_service_calls', 'state'].
2025-02-13 13:21:51.848 | INFO     | data.preprocessing.dataset_preprocessing:process_dataset:978 - X variables: Index(['account_length', 'international_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'area_code_0', 'area_code_1', 'area_code_2'],
      dtype='object').
```

# Usage: _Your_ $f_{\Omega}(X)$
`get_model(.)` in `utils/fe_models.py` is the entry point. It returns the $f_{\Omega}(X)$ (see Eq. 1 in the **MC-GMENN** [paper](https://www.ijcai.org/proceedings/2024/0555.pdf)) and an optimizer for that.

`model_name="simchoni_2021"` should make it equivalent to the model proposed in **LMMNN** [paper](https://arxiv.org/pdf/2206.03314).

`model_name="tabtransformer" should be a sensible choice.

You can roll out your model and add it to the `elif` ladder in `get_model(.)` – in order to tie it to your data.

# Usage: _Your_ $\phi(f_\Omega(\mathbf{X}) + \sum_{l} \mathbf{Z}^{(l)}\mathbf{B}^{(l)})$

Class `MixedEffectsNetwork` in `model/mixed_effects.py` is the entry point. It encapsulates the $\phi(f_\Omega(\mathbf{X}) + \sum_{l} \mathbf{Z}^{(l)}\mathbf{B}^{(l)})$ (see Eq. 1 in the **MC-GMENN** [paper](https://www.ijcai.org/proceedings/2024/0555.pdf)) and an optimizer for that.

My guess on the sensible defaults:
- `mode="intercepts"`
- `embed_x=False`
- `fe_pretraining=False`

_I am unsure_ about these defaults:-
- `fe_loss_weight=0.`
- `early_stopping_fe=None`

## Example: Processing `road-safety-drivers-sex`
```python
elif dataset_name == "road-safety-drivers-sex":
    # 1. Identify target
    y_col = "Sex_of_Driver"

    # 2. Identify binary columns = zwei Ausprägungen
    bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
    
    # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
    z_cols = list(
        df.nunique()[
            np.logical_and(df.nunique() >= hct, df.dtypes == "object")
        ].index
    )
        
    # 4. Identify cat cols = Rest dytpes==object
    cat_cols = list(
        set(df.dtypes[df.dtypes == "object"].index)
        - set([y_col] + bin_cols + z_cols)
    )
        
    # 5. Rest is numeric
    numeric_cols = list(
        set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols)
    )
    
    # 6. Label encode target
    le_ = LabelEncoder()
    df[y_col] = le_.fit_transform(df[y_col].astype(str))
```