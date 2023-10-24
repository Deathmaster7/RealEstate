{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b4f38e6e-74e4-43b9-90f4-83a6f1a3569b",
   "metadata": {},
   "source": [
    "## Dragon Real Estate Price Predictor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d60dbe8e-1615-4459-ab8a-b451dc0aecb0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1a68bbaf-5d4f-4a20-bbc2-40fdbdbbe37b",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing = pd.read_csv(\"data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5acf7158-5b7e-4109-a37d-92e88b531494",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1</td>\n",
       "      <td>296</td>\n",
       "      <td>15.3</td>\n",
       "      <td>396.90</td>\n",
       "      <td>4.98</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2</td>\n",
       "      <td>242</td>\n",
       "      <td>17.8</td>\n",
       "      <td>396.90</td>\n",
       "      <td>9.14</td>\n",
       "      <td>21.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2</td>\n",
       "      <td>242</td>\n",
       "      <td>17.8</td>\n",
       "      <td>392.83</td>\n",
       "      <td>4.03</td>\n",
       "      <td>34.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3</td>\n",
       "      <td>222</td>\n",
       "      <td>18.7</td>\n",
       "      <td>394.63</td>\n",
       "      <td>2.94</td>\n",
       "      <td>33.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3</td>\n",
       "      <td>222</td>\n",
       "      <td>18.7</td>\n",
       "      <td>396.90</td>\n",
       "      <td>5.33</td>\n",
       "      <td>36.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO  \\\n",
       "0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3   \n",
       "1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8   \n",
       "2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8   \n",
       "3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7   \n",
       "4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7   \n",
       "\n",
       "        B  LSTAT  MEDV  \n",
       "0  396.90   4.98  24.0  \n",
       "1  396.90   9.14  21.6  \n",
       "2  392.83   4.03  34.7  \n",
       "3  394.63   2.94  33.4  \n",
       "4  396.90   5.33  36.2  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8424a216-cebb-498f-a287-586f2a069b76",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 506 entries, 0 to 505\n",
      "Data columns (total 14 columns):\n",
      " #   Column   Non-Null Count  Dtype  \n",
      "---  ------   --------------  -----  \n",
      " 0   CRIM     506 non-null    float64\n",
      " 1   ZN       506 non-null    float64\n",
      " 2   INDUS    506 non-null    float64\n",
      " 3   CHAS     506 non-null    int64  \n",
      " 4   NOX      506 non-null    float64\n",
      " 5   RM       502 non-null    float64\n",
      " 6   AGE      506 non-null    float64\n",
      " 7   DIS      506 non-null    float64\n",
      " 8   RAD      506 non-null    int64  \n",
      " 9   TAX      506 non-null    int64  \n",
      " 10  PTRATIO  506 non-null    float64\n",
      " 11  B        506 non-null    float64\n",
      " 12  LSTAT    506 non-null    float64\n",
      " 13  MEDV     506 non-null    float64\n",
      "dtypes: float64(11), int64(3)\n",
      "memory usage: 55.5 KB\n"
     ]
    }
   ],
   "source": [
    "housing.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c34ef0a4-86ce-4ed4-91b8-abe404b7d976",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CHAS\n",
       "0    471\n",
       "1     35\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing['CHAS'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5b9414ad-b2a5-435c-a79e-1c8a5927c319",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>502.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.613524</td>\n",
       "      <td>11.363636</td>\n",
       "      <td>11.136779</td>\n",
       "      <td>0.069170</td>\n",
       "      <td>0.554695</td>\n",
       "      <td>6.285438</td>\n",
       "      <td>68.574901</td>\n",
       "      <td>3.795043</td>\n",
       "      <td>9.549407</td>\n",
       "      <td>408.237154</td>\n",
       "      <td>18.455534</td>\n",
       "      <td>356.674032</td>\n",
       "      <td>12.653063</td>\n",
       "      <td>22.532806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.601545</td>\n",
       "      <td>23.322453</td>\n",
       "      <td>6.860353</td>\n",
       "      <td>0.253994</td>\n",
       "      <td>0.115878</td>\n",
       "      <td>0.704633</td>\n",
       "      <td>28.148861</td>\n",
       "      <td>2.105710</td>\n",
       "      <td>8.707259</td>\n",
       "      <td>168.537116</td>\n",
       "      <td>2.164946</td>\n",
       "      <td>91.294864</td>\n",
       "      <td>7.141062</td>\n",
       "      <td>9.197104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.006320</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.460000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.385000</td>\n",
       "      <td>3.561000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>1.129600</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>187.000000</td>\n",
       "      <td>12.600000</td>\n",
       "      <td>0.320000</td>\n",
       "      <td>1.730000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.082045</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.190000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.449000</td>\n",
       "      <td>5.885500</td>\n",
       "      <td>45.025000</td>\n",
       "      <td>2.100175</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>279.000000</td>\n",
       "      <td>17.400000</td>\n",
       "      <td>375.377500</td>\n",
       "      <td>6.950000</td>\n",
       "      <td>17.025000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.256510</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.690000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.538000</td>\n",
       "      <td>6.209000</td>\n",
       "      <td>77.500000</td>\n",
       "      <td>3.207450</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>330.000000</td>\n",
       "      <td>19.050000</td>\n",
       "      <td>391.440000</td>\n",
       "      <td>11.360000</td>\n",
       "      <td>21.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.677083</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.624000</td>\n",
       "      <td>6.623500</td>\n",
       "      <td>94.075000</td>\n",
       "      <td>5.188425</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>666.000000</td>\n",
       "      <td>20.200000</td>\n",
       "      <td>396.225000</td>\n",
       "      <td>16.955000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>88.976200</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>27.740000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.871000</td>\n",
       "      <td>8.780000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>12.126500</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>711.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>396.900000</td>\n",
       "      <td>37.970000</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             CRIM          ZN       INDUS        CHAS         NOX          RM  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  502.000000   \n",
       "mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.285438   \n",
       "std      8.601545   23.322453    6.860353    0.253994    0.115878    0.704633   \n",
       "min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000   \n",
       "25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   \n",
       "50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.209000   \n",
       "75%      3.677083   12.500000   18.100000    0.000000    0.624000    6.623500   \n",
       "max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000   \n",
       "\n",
       "              AGE         DIS         RAD         TAX     PTRATIO           B  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   \n",
       "mean    68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   \n",
       "std     28.148861    2.105710    8.707259  168.537116    2.164946   91.294864   \n",
       "min      2.900000    1.129600    1.000000  187.000000   12.600000    0.320000   \n",
       "25%     45.025000    2.100175    4.000000  279.000000   17.400000  375.377500   \n",
       "50%     77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   \n",
       "75%     94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   \n",
       "max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   \n",
       "\n",
       "            LSTAT        MEDV  \n",
       "count  506.000000  506.000000  \n",
       "mean    12.653063   22.532806  \n",
       "std      7.141062    9.197104  \n",
       "min      1.730000    5.000000  \n",
       "25%      6.950000   17.025000  \n",
       "50%     11.360000   21.200000  \n",
       "75%     16.955000   25.000000  \n",
       "max     37.970000   50.000000  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "95265574-ea94-491f-9048-925ba36c7125",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8517e807-36c9-4f67-aba7-dd38e16432fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import matplotlib.pyplot as plt\n",
    "#housing.hist(bins=50, figsize=(20,15))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "071ff7dc-f9d8-4e61-a303-eedebf64aeb6",
   "metadata": {},
   "source": [
    "## Train-Test Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d9c29b6f-c915-41aa-818f-53098717327f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "def split_train_test(data, test_ratio):\n",
    "    np.random.seed(42)\n",
    "    shuffled = np.random.permutation(len(data))\n",
    "    test_set_size = int(len(data) * test_ratio)\n",
    "    test_indices = shuffled[:test_set_size]\n",
    "    train_indices = shuffled[test_set_size:]\n",
    "    return data.iloc[train_indices], data.iloc[test_indices]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "44cf48d7-da48-4729-bc1d-ecc40fda3bfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "#train_set, test_set = split_train_test(housing, 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "944bfb63-eef4-427a-a6f3-b3dbbed6b5d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(f\"Rows in train: {len(train_set)}\\nRows in test : {len(test_set)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8f9630b1-49b5-4c0a-9273-84f4c3e5804c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rows in train: 404\n",
      "Rows in test : 102\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)\n",
    "print(f\"Rows in train: {len(train_set)}\\nRows in test : {len(test_set)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3043e4c9-d484-4cec-870f-58761d93146d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state =42)\n",
    "for train_index, test_index in split.split(housing, housing['CHAS']):\n",
    "    strat_train_set = housing.loc[train_index]\n",
    "    strat_test_set = housing.loc[test_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b1cf22b5-ee5c-40ec-a9b2-2542f9572f5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CHAS\n",
       "0    376\n",
       "1     28\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strat_train_set['CHAS'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2d830f0e-6bda-4248-8465-7df87e671045",
   "metadata": {},
   "outputs": [],
   "source": [
    "#95/7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "94a38434-61b7-44ac-8b68-bd150035b5ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "#376/28"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "093d59d3-f4b1-43b7-bf9f-8d3c18d2c14a",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing = strat_train_set.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ff62906-4c3b-47e8-9096-9a3f19d807d8",
   "metadata": {},
   "source": [
    "## Looking For Correlations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "95a4092a-e1c9-4287-8e07-fce6fd438e20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MEDV       1.000000\n",
       "RM         0.679428\n",
       "B          0.361761\n",
       "ZN         0.339741\n",
       "DIS        0.240451\n",
       "CHAS       0.205066\n",
       "AGE       -0.364596\n",
       "RAD       -0.374693\n",
       "CRIM      -0.393715\n",
       "NOX       -0.422873\n",
       "TAX       -0.456657\n",
       "INDUS     -0.473516\n",
       "PTRATIO   -0.493534\n",
       "LSTAT     -0.740494\n",
       "Name: MEDV, dtype: float64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_matrix = housing.corr()\n",
    "corr_matrix['MEDV'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "df4eb21d-51c6-48cf-8db9-ea9815aaa090",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[<Axes: xlabel='RM', ylabel='RM'>,\n",
       "        <Axes: xlabel='ZN', ylabel='RM'>,\n",
       "        <Axes: xlabel='MEDV', ylabel='RM'>,\n",
       "        <Axes: xlabel='LSTAT', ylabel='RM'>],\n",
       "       [<Axes: xlabel='RM', ylabel='ZN'>,\n",
       "        <Axes: xlabel='ZN', ylabel='ZN'>,\n",
       "        <Axes: xlabel='MEDV', ylabel='ZN'>,\n",
       "        <Axes: xlabel='LSTAT', ylabel='ZN'>],\n",
       "       [<Axes: xlabel='RM', ylabel='MEDV'>,\n",
       "        <Axes: xlabel='ZN', ylabel='MEDV'>,\n",
       "        <Axes: xlabel='MEDV', ylabel='MEDV'>,\n",
       "        <Axes: xlabel='LSTAT', ylabel='MEDV'>],\n",
       "       [<Axes: xlabel='RM', ylabel='LSTAT'>,\n",
       "        <Axes: xlabel='ZN', ylabel='LSTAT'>,\n",
       "        <Axes: xlabel='MEDV', ylabel='LSTAT'>,\n",
       "        <Axes: xlabel='LSTAT', ylabel='LSTAT'>]], dtype=object)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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kMlkKCwsRGxsLHJzc7Fq1Srcfvvt2LZtGwDzFPfU1FROcyciIiJqxTXoRERkNZPJhKVLl0KpVOKDDz7odIVfJpPh1VdfxY4dO3D58mU7RUlERETk2DhAJyIiq23cuBE//PADDhw4AG9v7y6fM3v2bNxxxx3405/+NMjRERERETkHDtCJiMgq2dnZ2LJlC7KzszFy5Mgen/vqq6/iH//4B06dOjVI0RERERE5Dw7QiYhowE6fPo3Vq1fj1Vdf7VOJp0mTJuGBBx7Ahg0bBiE6IiIiIufCLO5ERDQglZWVSE1NxYwZM5CWltapjFN3JZ9efvlljB07Fkol/wURERERtcezIyIiGpB//vOfKCoqQlFRUacSTgAQHR2NnJycTtvj4uKwatUq7NixYxCiJCIiInIeLLNGRERERERE5AC4Bp2IiIiIiIjIAXCATkREREREROQAOEAnIiIiIiIicgAcoBMRERERERE5AA7QiYiIiIiIiBwAB+hEREREREREDoADdCIiIiIiIiIHwAE6ERERERERkQPgAJ2IiIiIiIjIAXCATkREREREROQAOEAnIiIiIiIicgAcoBMRERERERE5gP8HXG3AMzJD8MwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1200x800 with 16 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from pandas.plotting import scatter_matrix\n",
    "attributes = ['RM', \"ZN\", \"MEDV\", \"LSTAT\"]\n",
    "scatter_matrix(housing[attributes], figsize = (12,8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "9197ae4e-b969-417a-95ab-8c9b02a0877f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='RM', ylabel='MEDV'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "housing.plot(kind=\"scatter\", x=\"RM\", y=\"MEDV\", alpha=0.8)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af7de792-45a3-40e9-b7b3-95e6e4cd9978",
   "metadata": {},
   "source": [
    "## Trying Out Attribute Combinations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fbbf2909-1ab1-4257-b343-699160615e38",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing[\"TAXRM\"] = housing[\"TAX\"]/housing[\"RM\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "55f19d87-4185-4177-aba8-26e3e708d27f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "254     51.571709\n",
       "348     42.200452\n",
       "476    102.714374\n",
       "321     45.012547\n",
       "326     45.468948\n",
       "          ...    \n",
       "155     65.507152\n",
       "423    109.126659\n",
       "98      35.294118\n",
       "455    102.068966\n",
       "216     46.875000\n",
       "Name: TAXRM, Length: 404, dtype: float64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing[\"TAXRM\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "619d678e-d151-43b0-9959-fae44cf2553e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "      <th>TAXRM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>254</th>\n",
       "      <td>0.04819</td>\n",
       "      <td>80.0</td>\n",
       "      <td>3.64</td>\n",
       "      <td>0</td>\n",
       "      <td>0.392</td>\n",
       "      <td>6.108</td>\n",
       "      <td>32.0</td>\n",
       "      <td>9.2203</td>\n",
       "      <td>1</td>\n",
       "      <td>315</td>\n",
       "      <td>16.4</td>\n",
       "      <td>392.89</td>\n",
       "      <td>6.57</td>\n",
       "      <td>21.9</td>\n",
       "      <td>51.571709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>348</th>\n",
       "      <td>0.01501</td>\n",
       "      <td>80.0</td>\n",
       "      <td>2.01</td>\n",
       "      <td>0</td>\n",
       "      <td>0.435</td>\n",
       "      <td>6.635</td>\n",
       "      <td>29.7</td>\n",
       "      <td>8.3440</td>\n",
       "      <td>4</td>\n",
       "      <td>280</td>\n",
       "      <td>17.0</td>\n",
       "      <td>390.94</td>\n",
       "      <td>5.99</td>\n",
       "      <td>24.5</td>\n",
       "      <td>42.200452</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>476</th>\n",
       "      <td>4.87141</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.614</td>\n",
       "      <td>6.484</td>\n",
       "      <td>93.6</td>\n",
       "      <td>2.3053</td>\n",
       "      <td>24</td>\n",
       "      <td>666</td>\n",
       "      <td>20.2</td>\n",
       "      <td>396.21</td>\n",
       "      <td>18.68</td>\n",
       "      <td>16.7</td>\n",
       "      <td>102.714374</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>321</th>\n",
       "      <td>0.18159</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.376</td>\n",
       "      <td>54.3</td>\n",
       "      <td>4.5404</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.87</td>\n",
       "      <td>23.1</td>\n",
       "      <td>45.012547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>326</th>\n",
       "      <td>0.30347</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.312</td>\n",
       "      <td>28.9</td>\n",
       "      <td>5.4159</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.15</td>\n",
       "      <td>23.0</td>\n",
       "      <td>45.468948</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  \\\n",
       "254  0.04819  80.0   3.64     0  0.392  6.108  32.0  9.2203    1  315   \n",
       "348  0.01501  80.0   2.01     0  0.435  6.635  29.7  8.3440    4  280   \n",
       "476  4.87141   0.0  18.10     0  0.614  6.484  93.6  2.3053   24  666   \n",
       "321  0.18159   0.0   7.38     0  0.493  6.376  54.3  4.5404    5  287   \n",
       "326  0.30347   0.0   7.38     0  0.493  6.312  28.9  5.4159    5  287   \n",
       "\n",
       "     PTRATIO       B  LSTAT  MEDV       TAXRM  \n",
       "254     16.4  392.89   6.57  21.9   51.571709  \n",
       "348     17.0  390.94   5.99  24.5   42.200452  \n",
       "476     20.2  396.21  18.68  16.7  102.714374  \n",
       "321     19.6  396.90   6.87  23.1   45.012547  \n",
       "326     19.6  396.90   6.15  23.0   45.468948  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "762bcf2c-1b46-49ef-8d7c-3fbea50f9564",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MEDV       1.000000\n",
       "RM         0.679428\n",
       "B          0.361761\n",
       "ZN         0.339741\n",
       "DIS        0.240451\n",
       "CHAS       0.205066\n",
       "AGE       -0.364596\n",
       "RAD       -0.374693\n",
       "CRIM      -0.393715\n",
       "NOX       -0.422873\n",
       "TAX       -0.456657\n",
       "INDUS     -0.473516\n",
       "PTRATIO   -0.493534\n",
       "TAXRM     -0.526668\n",
       "LSTAT     -0.740494\n",
       "Name: MEDV, dtype: float64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_matrix = housing.corr()\n",
    "corr_matrix['MEDV'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f3290a58-044d-4388-938d-0065ad7f4f8d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='TAXRM', ylabel='MEDV'>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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rlyJn/3sZ3jooYewefNm1NTU4OKLL4aqqiU88uH94Ac/wIMPPohf/OIX2LNnD37wgx9g5cqV+PnPf564TyWNLRwO48wzz8Qvf/nLtLdnM5Zrr70Wu3btwksvvYTnnnsOGzZswM033zxaQ8go0/gikQjefPNNrFixAm+++SZWr16Nzs5OXH755Sn3K9fxDffexT399NPYtGkT2tvbB91WrmMDhh/fe++9h49+9KOYNWsWXnnlFbz11ltYsWIF3G534j7leh4dbmx33nkn1q5di//+7//Gnj17cPvtt+O2227DM888k7hPuY5tWHwMO3r0KAfA169fzznn3OfzcVmW+VNPPZW4z549ezgAvnHjxlIdZk6CwSA/9dRT+UsvvcTPP/98/qUvfYlzXvlj+9rXvsY/+tGPDnm7bdu8ra2N//CHP0z8zOfzcZfLxZ944onROMS8XXLJJfyGG25I+dlVV13Fr732Ws55ZY8NAH/66acTf89mLLt37+YA+N/+9rfEfV544QXOGONdXV2jduzZGDi+dN544w0OgO/fv59zXjnjG2pshw4d4h0dHXznzp186tSp/Mc//nHitkoZG+fpx/fpT3+aX3fddUP+TqWcR9ONbc6cOfzb3/52ys/OPvts/o1vfINzXjljS2dMZWQG8vv9AIDm5mYAwNatW2EYBi666KLEfWbNmoUpU6Zg48aNJTnGXC1fvhyXXHJJyhiAyh/bM888gwULFuBTn/oUJkyYgHnz5uHXv/514vZ9+/ahu7s7ZXwNDQ0455xzyn58H/nIR7Bu3Trs3bsXAPD3v/8dr776KpYuXQqgssc2UDZj2bhxIxobG7FgwYLEfS666CIIgoDNmzeP+jGPlN/vB2MMjY2NACp7fLZtY9myZfjqV7+KOXPmDLq90sf2/PPPY8aMGbj44osxYcIEnHPOOSlTNJV8Hv3IRz6CZ555Bl1dXeCc4+WXX8bevXuxePFiAJU9tjEbyNi2jdtvvx3nnnsu5s6dCwDo7u6GoiiJE05ca2sruru7S3CUuXnyySfx5ptv4v777x90W6WP7f3338eDDz6IU089FX/605/whS98Af/6r/+KRx99FAASY2htbU35vUoY37/927/hM5/5DGbNmgVZljFv3jzcfvvtuPbaawFU9tgGymYs3d3dmDBhQsrtkiShubm54sarqiq+9rWv4Zpr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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "housing.plot(kind=\"scatter\", x=\"TAXRM\", y=\"MEDV\", alpha=0.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4df6457b-e57a-45e1-854a-f28167d0507e",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing = strat_train_set.drop(\"MEDV\", axis=1)\n",
    "housing_labels = strat_train_set[\"MEDV\"].copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c2644c3-4129-41b1-9ff8-8aec59100124",
   "metadata": {},
   "source": [
    "## Missing Attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "25951c7f-95a0-4855-b5a3-581b024dd15d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#To take care of missing attributes, you have threee options:\n",
    " #   1.Get rid of the missing data points\n",
    "  #  2.Get rid of whole attribute\n",
    "   # 3.Set the value to some value(0, mean or median)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "b9c701cc-caa6-4b00-9e06-e688d5039881",
   "metadata": {},
   "outputs": [],
   "source": [
    "a = housing.dropna(subset=[\"RM\"]) # Option 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "f6fc009e-8286-4f85-aa30-a26e87a7cbaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(400, 13)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d79090a8-c2a1-4397-8d06-9d57825042ec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(404, 12)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.drop(\"RM\", axis=1).shape #Option 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e9f1ed5b-7486-4d41-9d6d-4769930d802f",
   "metadata": {},
   "outputs": [],
   "source": [
    "median = housing[\"RM\"].median() #Option 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "fa627375-6437-4080-8c90-f393acc729c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "254    6.108\n",
       "348    6.635\n",
       "476    6.484\n",
       "321    6.376\n",
       "326    6.312\n",
       "       ...  \n",
       "155    6.152\n",
       "423    6.103\n",
       "98     7.820\n",
       "455    6.525\n",
       "216    5.888\n",
       "Name: RM, Length: 404, dtype: float64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing[\"RM\"].fillna(median)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "643e21d3-0f1e-4dd8-a7f1-e301870438aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.602814</td>\n",
       "      <td>10.836634</td>\n",
       "      <td>11.344950</td>\n",
       "      <td>0.069307</td>\n",
       "      <td>0.558064</td>\n",
       "      <td>6.280870</td>\n",
       "      <td>69.039851</td>\n",
       "      <td>3.746210</td>\n",
       "      <td>9.735149</td>\n",
       "      <td>412.341584</td>\n",
       "      <td>18.473267</td>\n",
       "      <td>353.392822</td>\n",
       "      <td>12.791609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.099383</td>\n",
       "      <td>22.150636</td>\n",
       "      <td>6.877817</td>\n",
       "      <td>0.254290</td>\n",
       "      <td>0.116875</td>\n",
       "      <td>0.715586</td>\n",
       "      <td>28.258248</td>\n",
       "      <td>2.099057</td>\n",
       "      <td>8.731259</td>\n",
       "      <td>168.672623</td>\n",
       "      <td>2.129243</td>\n",
       "      <td>96.069235</td>\n",
       "      <td>7.235740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.006320</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.740000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.389000</td>\n",
       "      <td>3.561000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>1.129600</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>187.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>0.320000</td>\n",
       "      <td>1.730000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.086962</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.190000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.453000</td>\n",
       "      <td>5.878750</td>\n",
       "      <td>44.850000</td>\n",
       "      <td>2.035975</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>284.000000</td>\n",
       "      <td>17.400000</td>\n",
       "      <td>374.617500</td>\n",
       "      <td>6.847500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.286735</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.900000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.538000</td>\n",
       "      <td>6.213500</td>\n",
       "      <td>78.200000</td>\n",
       "      <td>3.122200</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>337.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>390.955000</td>\n",
       "      <td>11.570000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.731923</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.631000</td>\n",
       "      <td>6.630250</td>\n",
       "      <td>94.100000</td>\n",
       "      <td>5.100400</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>666.000000</td>\n",
       "      <td>20.200000</td>\n",
       "      <td>395.630000</td>\n",
       "      <td>17.102500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>73.534100</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>27.740000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.871000</td>\n",
       "      <td>8.780000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>12.126500</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>711.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>396.900000</td>\n",
       "      <td>36.980000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             CRIM          ZN       INDUS        CHAS         NOX          RM  \\\n",
       "count  404.000000  404.000000  404.000000  404.000000  404.000000  400.000000   \n",
       "mean     3.602814   10.836634   11.344950    0.069307    0.558064    6.280870   \n",
       "std      8.099383   22.150636    6.877817    0.254290    0.116875    0.715586   \n",
       "min      0.006320    0.000000    0.740000    0.000000    0.389000    3.561000   \n",
       "25%      0.086962    0.000000    5.190000    0.000000    0.453000    5.878750   \n",
       "50%      0.286735    0.000000    9.900000    0.000000    0.538000    6.213500   \n",
       "75%      3.731923   12.500000   18.100000    0.000000    0.631000    6.630250   \n",
       "max     73.534100  100.000000   27.740000    1.000000    0.871000    8.780000   \n",
       "\n",
       "              AGE         DIS         RAD         TAX     PTRATIO           B  \\\n",
       "count  404.000000  404.000000  404.000000  404.000000  404.000000  404.000000   \n",
       "mean    69.039851    3.746210    9.735149  412.341584   18.473267  353.392822   \n",
       "std     28.258248    2.099057    8.731259  168.672623    2.129243   96.069235   \n",
       "min      2.900000    1.129600    1.000000  187.000000   13.000000    0.320000   \n",
       "25%     44.850000    2.035975    4.000000  284.000000   17.400000  374.617500   \n",
       "50%     78.200000    3.122200    5.000000  337.000000   19.000000  390.955000   \n",
       "75%     94.100000    5.100400   24.000000  666.000000   20.200000  395.630000   \n",
       "max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   \n",
       "\n",
       "            LSTAT  \n",
       "count  404.000000  \n",
       "mean    12.791609  \n",
       "std      7.235740  \n",
       "min      1.730000  \n",
       "25%      6.847500  \n",
       "50%     11.570000  \n",
       "75%     17.102500  \n",
       "max     36.980000  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "27612a8d-4093-4152-bc5d-d99f0116a88b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "SimpleImputer(strategy='median')"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy = \"median\")\n",
    "imputer.fit(housing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a07a286c-72e7-45aa-b31b-aa5460c55aaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2.86735e-01, 0.00000e+00, 9.90000e+00, 0.00000e+00, 5.38000e-01,\n",
       "       6.21350e+00, 7.82000e+01, 3.12220e+00, 5.00000e+00, 3.37000e+02,\n",
       "       1.90000e+01, 3.90955e+02, 1.15700e+01])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imputer.statistics_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "8a100105-900c-4063-8b43-35449a93712d",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = imputer.transform(housing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "26f9f9e3-eb3a-4127-b42d-e7733c698168",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing_tr = pd.DataFrame(x, columns=housing.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "43089596-26eb-4153-97f2-dc266addf5e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "      <td>404.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.602814</td>\n",
       "      <td>10.836634</td>\n",
       "      <td>11.344950</td>\n",
       "      <td>0.069307</td>\n",
       "      <td>0.558064</td>\n",
       "      <td>6.280203</td>\n",
       "      <td>69.039851</td>\n",
       "      <td>3.746210</td>\n",
       "      <td>9.735149</td>\n",
       "      <td>412.341584</td>\n",
       "      <td>18.473267</td>\n",
       "      <td>353.392822</td>\n",
       "      <td>12.791609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.099383</td>\n",
       "      <td>22.150636</td>\n",
       "      <td>6.877817</td>\n",
       "      <td>0.254290</td>\n",
       "      <td>0.116875</td>\n",
       "      <td>0.712057</td>\n",
       "      <td>28.258248</td>\n",
       "      <td>2.099057</td>\n",
       "      <td>8.731259</td>\n",
       "      <td>168.672623</td>\n",
       "      <td>2.129243</td>\n",
       "      <td>96.069235</td>\n",
       "      <td>7.235740</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.006320</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.740000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.389000</td>\n",
       "      <td>3.561000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>1.129600</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>187.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>0.320000</td>\n",
       "      <td>1.730000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.086962</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.190000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.453000</td>\n",
       "      <td>5.879750</td>\n",
       "      <td>44.850000</td>\n",
       "      <td>2.035975</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>284.000000</td>\n",
       "      <td>17.400000</td>\n",
       "      <td>374.617500</td>\n",
       "      <td>6.847500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.286735</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.900000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.538000</td>\n",
       "      <td>6.213500</td>\n",
       "      <td>78.200000</td>\n",
       "      <td>3.122200</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>337.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>390.955000</td>\n",
       "      <td>11.570000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.731923</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.631000</td>\n",
       "      <td>6.630000</td>\n",
       "      <td>94.100000</td>\n",
       "      <td>5.100400</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>666.000000</td>\n",
       "      <td>20.200000</td>\n",
       "      <td>395.630000</td>\n",
       "      <td>17.102500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>73.534100</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>27.740000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.871000</td>\n",
       "      <td>8.780000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>12.126500</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>711.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>396.900000</td>\n",
       "      <td>36.980000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             CRIM          ZN       INDUS        CHAS         NOX          RM  \\\n",
       "count  404.000000  404.000000  404.000000  404.000000  404.000000  404.000000   \n",
       "mean     3.602814   10.836634   11.344950    0.069307    0.558064    6.280203   \n",
       "std      8.099383   22.150636    6.877817    0.254290    0.116875    0.712057   \n",
       "min      0.006320    0.000000    0.740000    0.000000    0.389000    3.561000   \n",
       "25%      0.086962    0.000000    5.190000    0.000000    0.453000    5.879750   \n",
       "50%      0.286735    0.000000    9.900000    0.000000    0.538000    6.213500   \n",
       "75%      3.731923   12.500000   18.100000    0.000000    0.631000    6.630000   \n",
       "max     73.534100  100.000000   27.740000    1.000000    0.871000    8.780000   \n",
       "\n",
       "              AGE         DIS         RAD         TAX     PTRATIO           B  \\\n",
       "count  404.000000  404.000000  404.000000  404.000000  404.000000  404.000000   \n",
       "mean    69.039851    3.746210    9.735149  412.341584   18.473267  353.392822   \n",
       "std     28.258248    2.099057    8.731259  168.672623    2.129243   96.069235   \n",
       "min      2.900000    1.129600    1.000000  187.000000   13.000000    0.320000   \n",
       "25%     44.850000    2.035975    4.000000  284.000000   17.400000  374.617500   \n",
       "50%     78.200000    3.122200    5.000000  337.000000   19.000000  390.955000   \n",
       "75%     94.100000    5.100400   24.000000  666.000000   20.200000  395.630000   \n",
       "max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   \n",
       "\n",
       "            LSTAT  \n",
       "count  404.000000  \n",
       "mean    12.791609  \n",
       "std      7.235740  \n",
       "min      1.730000  \n",
       "25%      6.847500  \n",
       "50%     11.570000  \n",
       "75%     17.102500  \n",
       "max     36.980000  "
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing_tr.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06240a98-12de-4efc-9a92-da7cf74e34a9",
   "metadata": {},
   "source": [
    "## Scikit-learn Design"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f148e7f-5c6e-4d65-a6cd-a4fd2fd849da",
   "metadata": {},
   "source": [
    "Primarily, three types of objects\n",
    "1. Estimators - \n",
    "2. Transformers\n",
    "3. Predictors"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f6eab97-2335-4fcb-9f14-8668c9514c27",
   "metadata": {},
   "source": [
    "## Feature Scaling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2688415-ea21-4a3f-8410-c13a17bfe746",
   "metadata": {},
   "source": [
    "Primarily two types of feature scaling methods:\n",
    "1. Min-Max scaling (Normalization)\n",
    "   (value-min)/(max-min)\n",
    "   Sklearn provides a class called MinMaxScaler for this\n",
    "3. Standardization\n",
    "   (value-mean)/std\n",
    "   Sklearn provides a class called Standard Scaler for this"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "072a7355-b85f-446a-9336-e28b09decd38",
   "metadata": {},
   "source": [
    "## Creating Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3b686828-4a2e-4036-bfd2-11190e97896e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "my_pipeline=Pipeline([\n",
    "    ('imputer' , SimpleImputer(strategy='median')),\n",
    "    ('std_scaler', StandardScaler())\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b312cb92-993b-472f-94e1-b014dd1abd35",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing_num_tr = my_pipeline.fit_transform(housing)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab632f34-00e1-4e68-8076-b0d62a1756c8",
   "metadata": {},
   "source": [
    "## Selecting a desired model for realestate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "aae97384-a23c-4ef9-aab0-2663a93ba112",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "#model = LinearRegression()\n",
    "#model = DecisionTreeRegressor()\n",
    "model = RandomForestRegressor()\n",
    "model.fit(housing_num_tr, housing_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "14e71329-b622-4765-97ac-7ac4d0cc592a",
   "metadata": {},
   "outputs": [],
   "source": [
    "some_data = housing.iloc[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "5c28830c-2ae5-4047-bd60-a5015a1085b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "some_labels = housing_labels.iloc[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "2feda6cb-f2a4-475a-afbb-f7c0a941821f",
   "metadata": {},
   "outputs": [],
   "source": [
    "prepared_data = my_pipeline.transform(some_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "1a515088-5bea-46b1-b878-14c8d1f69a33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([22.254, 25.636, 16.436, 23.446, 23.506])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(prepared_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "cea8a327-1be0-4456-910c-0d300e31fe01",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[21.9, 24.5, 16.7, 23.1, 23.0]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(some_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0e9f3f7-93d3-4bf5-9352-55f9a01b5182",
   "metadata": {},
   "source": [
    "## Evaluating the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "8cf55f2e-914d-4ed8-830d-dd7b3e616b22",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "housing_predictions = model.predict(housing_num_tr)\n",
    "mse = mean_squared_error(housing_labels, housing_predictions)\n",
    "rmse = np.sqrt(mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "bc3940d3-0763-44c3-9371-8740d8274635",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.6640948564356424"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a34cf6fc-7f8b-4ceb-832f-cc4d6babddee",
   "metadata": {},
   "source": [
    "## Using better evaluation technique - Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b2f10ac2-879e-4d2b-95c5-634865fa5c10",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1 2 3 4 5 6 7 8 9 10\n",
    "from sklearn.model_selection import cross_val_score\n",
    "scores = cross_val_score(model, housing_num_tr, housing_labels, scoring=\"neg_mean_squared_error\", cv=10)\n",
    "rmse_scores = np.sqrt(-scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "d147183a-648f-4362-b327-323aeaf35f19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2.92833974, 2.71424559, 4.4114325 , 2.47261594, 3.52833564,\n",
       "       2.4908109 , 4.84938371, 3.35358708, 3.16134592, 3.27559952])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rmse_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "615bb1fb-66e5-4fa0-9e7d-b588db7d8c8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_scores(scores):\n",
    "    print(\"Scores\", scores)\n",
    "    print(\"Mean\", scores.mean())\n",
    "    print(\"Standard Deviation\", scores.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "da5c2ad0-1d49-4c12-bb63-7fbac8b3a61a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Scores [2.92833974 2.71424559 4.4114325  2.47261594 3.52833564 2.4908109\n",
      " 4.84938371 3.35358708 3.16134592 3.27559952]\n",
      "Mean 3.3185696540339586\n",
      "Standard Deviation 0.7434947082548587\n"
     ]
    }
   ],
   "source": [
    "print_scores(rmse_scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7a2bef3-f64c-459d-8120-982fd09e3f28",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
