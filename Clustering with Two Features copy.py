{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "241ff9ce-bd6c-4949-9206-e98982e9bbaf",
   "metadata": {
    "tags": []
   },
   "source": [
    "<font size=\"+3\"><strong>Clustering with Two Features</strong></font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c314e461-4968-4d56-a3b1-7bb9f4d1b141",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e118d14-d5d1-4b76-9cf2-d05f4aaf37f1",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Prepare Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "527236cc-02b7-4815-b788-820064f3bb91",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Import\n",
    "\n",
    "Just like always, we need to begin by bringing our data into the project. We spent some time in the previous lesson working with a subset of the larger SCF dataset called `\"TURNFEAR\"`. Let's start with that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3fc9712c-73c5-4432-914c-047050251aaf",
   "metadata": {
    "deletable": false
   },
   "outputs": [],
   "source": [
    "def wrangle(filepath):\n",
    "    df = pd.read_csv(filepath)\n",
    "    mask = df[\"TURNFEAR\"] == 1\n",
    "    df = df[mask]\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "aa8e58d2-ddbd-4a50-b2aa-8ece0952748c",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4623, 351)\n"
     ]
    },
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
       "      <th>YY1</th>\n",
       "      <th>Y1</th>\n",
       "      <th>WGT</th>\n",
       "      <th>HHSEX</th>\n",
       "      <th>AGE</th>\n",
       "      <th>AGECL</th>\n",
       "      <th>EDUC</th>\n",
       "      <th>EDCL</th>\n",
       "      <th>MARRIED</th>\n",
       "      <th>KIDS</th>\n",
       "      <th>...</th>\n",
       "      <th>NWCAT</th>\n",
       "      <th>INCCAT</th>\n",
       "      <th>ASSETCAT</th>\n",
       "      <th>NINCCAT</th>\n",
       "      <th>NINC2CAT</th>\n",
       "      <th>NWPCTLECAT</th>\n",
       "      <th>INCPCTLECAT</th>\n",
       "      <th>NINCPCTLECAT</th>\n",
       "      <th>INCQRTCAT</th>\n",
       "      <th>NINCQRTCAT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2</td>\n",
       "      <td>21</td>\n",
       "      <td>3790.476607</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2</td>\n",
       "      <td>22</td>\n",
       "      <td>3798.868505</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2</td>\n",
       "      <td>23</td>\n",
       "      <td>3799.468393</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2</td>\n",
       "      <td>24</td>\n",
       "      <td>3788.076005</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2</td>\n",
       "      <td>25</td>\n",
       "      <td>3793.066589</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>3</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 351 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   YY1  Y1          WGT  HHSEX  AGE  AGECL  EDUC  EDCL  MARRIED  KIDS  ...  \\\n",
       "5    2  21  3790.476607      1   50      3     8     2        1     3  ...   \n",
       "6    2  22  3798.868505      1   50      3     8     2        1     3  ...   \n",
       "7    2  23  3799.468393      1   50      3     8     2        1     3  ...   \n",
       "8    2  24  3788.076005      1   50      3     8     2        1     3  ...   \n",
       "9    2  25  3793.066589      1   50      3     8     2        1     3  ...   \n",
       "\n",
       "   NWCAT  INCCAT  ASSETCAT  NINCCAT  NINC2CAT  NWPCTLECAT  INCPCTLECAT  \\\n",
       "5      1       2         1        2         1           1            4   \n",
       "6      1       2         1        2         1           1            4   \n",
       "7      1       2         1        2         1           1            4   \n",
       "8      1       2         1        2         1           1            4   \n",
       "9      1       2         1        2         1           1            4   \n",
       "\n",
       "   NINCPCTLECAT  INCQRTCAT  NINCQRTCAT  \n",
       "5             4          2           2  \n",
       "6             3          2           2  \n",
       "7             4          2           2  \n",
       "8             4          2           2  \n",
       "9             4          2           2  \n",
       "\n",
       "[5 rows x 351 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = wrangle(\"SCFP2019.csv\")\n",
    "print(df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a3dfb8f-b266-47b1-86bc-ecd99bdc2340",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Explore"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b63fc323-9eda-434b-91f7-1b44b06264ec",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot \"HOUSES\" vs \"DEBT\"\n",
    "sns.scatterplot(x=df[\"DEBT\"]/1e6, y=df[\"HOUSES\"]/1e6)\n",
    "plt.xlabel(\"Household Debt [$1M]\")\n",
    "plt.ylabel(\"Home Value [$1M]\")\n",
    "plt.title(\"Credit Fearful: Home Value vs. Household Debt\");"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99c7c89f-3c23-430d-89bc-6c84fbeda093",
   "metadata": {
    "tags": []
   },
   "source": [
    "Remember that graph and its clusters? Let's get a little deeper into it."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8c6e172-247c-41eb-938c-e6ea962078ec",
   "metadata": {
    "deletable": false,
    "editable": false,
    "tags": []
   },
   "source": [
    "## Split"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd1aea0d-66e2-4172-ab54-522e5bdc6d7e",
   "metadata": {
    "tags": []
   },
   "source": [
    "We need to split our data, but we're not going to need target vector or a test set this time around. That's because the model we'll be building involves *unsupervised* learning. It's called *unsupervised* because the model doesn't try to map input to a st of labels or targets that already exist. It's kind of like how humans learn new skills, in that we don't always have models to copy. Sometimes, we just try out something and see what happens. Keep in mind that this doesn't make these models any less useful, it just makes them different.\n",
    "\n",
    "So, keeping that in mind, let's do the split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1eef1531-d919-4a85-8eaa-148750c2d5c0",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4623, 2)\n"
     ]
    },
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
       "      <th>DEBT</th>\n",
       "      <th>HOUSES</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>12200.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>12600.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>15300.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>14100.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>15400.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      DEBT  HOUSES\n",
       "5  12200.0     0.0\n",
       "6  12600.0     0.0\n",
       "7  15300.0     0.0\n",
       "8  14100.0     0.0\n",
       "9  15400.0     0.0"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[[\"DEBT\", \"HOUSES\"]]\n",
    "print(X.shape)\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd9ae9a4-a490-4a15-b300-4b3b5c8109a5",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Build Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "11a542d2-518f-433b-be05-7e9f5c8ceeec",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'ClusterWidget' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [19]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m cw \u001b[38;5;241m=\u001b[39m \u001b[43mClusterWidget\u001b[49m(n_clusters\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m)\n\u001b[0;32m      2\u001b[0m cw\u001b[38;5;241m.\u001b[39mshow()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'ClusterWidget' is not defined"
     ]
    }
   ],
   "source": [
    "cw = ClusterWidget(n_clusters=3)\n",
    "cw.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1dd60210-5bfa-4c96-b96d-c49d3ed77f6a",
   "metadata": {
    "tags": []
   },
   "source": [
    "Since a centroid represents the mean value of all the data in the cluster, we would expect it to fall in the center of whatever cluster it's in. That's what will happen if you move the slider one more position to the right. See how the centroids moved? "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e555382f-0702-4f3a-9863-9ab8cff8687f",
   "metadata": {
    "deletable": false,
    "tags": []
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SCFClusterWidget' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [20]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m scfc \u001b[38;5;241m=\u001b[39m \u001b[43mSCFClusterWidget\u001b[49m(x\u001b[38;5;241m=\u001b[39mdf[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDEBT\u001b[39m\u001b[38;5;124m\"\u001b[39m], y\u001b[38;5;241m=\u001b[39mdf[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mHOUSES\u001b[39m\u001b[38;5;124m\"\u001b[39m], n_clusters\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m)\n\u001b[0;32m      2\u001b[0m scfc\u001b[38;5;241m.\u001b[39mshow()\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SCFClusterWidget' is not defined"
     ]
    }
   ],
   "source": [
    "scfc = SCFClusterWidget(x=df[\"DEBT\"], y=df[\"HOUSES\"], n_clusters=3)\n",
    "scfc.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9a916f0-d3b9-43dc-8cc6-0a838757b672",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Iterate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4426994a-a944-4e4f-903d-62cf639a14e8",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KMeans(n_clusters=3, random_state=42)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Build model\n",
    "model = KMeans(n_clusters=3, random_state=42)\n",
    "# Fit model to data\n",
    "model.fit(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "933b266a-94a1-45e1-b87c-3e934753c5ab",
   "metadata": {
    "tags": []
   },
   "source": [
    "And there it is. 42 datapoints spread across three clusters. Let's grab the labels that the model has assigned to the data points so we can start making a new visualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cd5fbe19-b0c5-44eb-84c7-942c08c4280b",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "labels = model.labels_\n",
    "labels[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bde33cf2-8576-48df-8dad-9512584edeae",
   "metadata": {
    "tags": []
   },
   "source": [
    "Using the labels we just extracted, let's recreate the scatter plot from before, this time we'll color each point according to the cluster to which the model assigned it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "d9994a61-c6a3-4ccd-9714-edee0ccf2c62",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot \"HOUSES\" vs \"DEBT\" with hue=label\n",
    "sns.scatterplot(\n",
    "    x=df[\"DEBT\"]/1e6,\n",
    "    y=df[\"HOUSES\"]/1e6,\n",
    "    hue=labels,\n",
    "    palette=\"deep\"\n",
    ")\n",
    "plt.xlabel(\"Household Debt [$1M]\")\n",
    "plt.ylabel(\"Home Value [$1M]\")\n",
    "plt.title(\"Credit Fearful: Home Value vs. Household Debt\");"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b448021-405c-4422-8670-ce5a51832c5d",
   "metadata": {
    "tags": []
   },
   "source": [
    "Nice! Each cluster has its own color. The centroids are still missing, so let's pull those out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "bd2340a9-8f69-4ce4-8fca-d80718bfd9e2",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[   91017.57766674,   116150.29328699],\n",
       "       [18384100.        , 34484000.        ],\n",
       "       [ 5065800.        , 11666666.66666667]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "centroids = model.cluster_centers_\n",
    "centroids"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a62afc8-9119-4ea9-a4ba-c481692e3d9b",
   "metadata": {
    "tags": []
   },
   "source": [
    "Let's add the centroids to the graph."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e646d42b-2b5f-44e6-b349-e37c29621540",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot \"HOUSES\" vs \"DEBT\", add centroids\n",
    "sns.scatterplot(\n",
    "    x=df[\"DEBT\"]/1e6,\n",
    "    y=df[\"HOUSES\"]/1e6,\n",
    "    hue=labels,\n",
    "    palette=\"deep\"\n",
    ")\n",
    "plt.scatter(\n",
    "    x=centroids[:, 0]/1e6,\n",
    "    y=centroids[:, 1]/1e6,\n",
    "    color=\"gray\",\n",
    "    marker=\"*\",\n",
    "    s=150\n",
    ")\n",
    "plt.xlabel(\"Household Debt [$1M]\")\n",
    "plt.ylabel(\"Home Value [$1M]\")\n",
    "plt.title(\"Credit Fearful: Home Value vs. Household Debt\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e606a8e4-8087-4f9f-ab9f-def5a2c02e17",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Inertia (3 clusters): 939554010797047.9\n"
     ]
    }
   ],
   "source": [
    "inertia = model.inertia_\n",
    "print(\"Inertia (3 clusters):\", inertia)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5a860be-2cd4-4d40-9285-4748ba2222a3",
   "metadata": {
    "tags": []
   },
   "source": [
    "The \"best\" inertia is 0, and our score is pretty far from that. Does that mean our model is \"bad?\" Not necessarily. Inertia is a measurement of distance (like mean absolute error from Project 2). This means that the unit of measurement for inertia depends on the unit of measurement of our x- and y-axes. And since `\"DEBT\"` and `\"HOUSES\"` are measured in tens of millions of dollars, it's not surprising that inertia is so large. \n",
    "\n",
    "However, it would be helpful to have metric that was easier to interpret, and that's where **silhouette score** comes in. Silhouette score measures the distance *between different clusters*. It ranges from -1 (the worst) to 1 (the best), so it's easier to interpret than inertia. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "28ee87d9-a855-4f84-b998-754e64330972",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Silhouette Score (3 clusters): 0.9768842462944348\n"
     ]
    }
   ],
   "source": [
    "ss = silhouette_score(X, model.labels_)\n",
    "print(\"Silhouette Score (3 clusters):\", ss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7161b0d6-4ceb-4576-b541-0e58421f0a00",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Inertia: [3018038313336915.5, 939554010797047.9, 546098841715653.0, 309310386410919.4, 235243397481788.4, 182225729179699.5, 150670779013795.16, 114321995931020.88, 100340259483916.77, 86229997033602.4, 74757234072100.31]\n",
      "\n",
      "Silhouette Scores: [0.9855099957519555, 0.9768842462944348, 0.9490311483406091, 0.839330043242819, 0.7287406719898627, 0.726989114305748, 0.7263840026889208, 0.7335125606476427, 0.692157992955073, 0.6949309528556856, 0.6951831031001252]\n"
     ]
    }
   ],
   "source": [
    "n_clusters = range(2, 13)\n",
    "inertia_errors = []\n",
    "silhouette_scores = [] \n",
    "\n",
    "# Add `for` loop to train model and calculate inertia, silhouette score.\n",
    "for k in n_clusters:\n",
    "    model = KMeans(n_clusters=k, random_state=42)\n",
    "    model.fit(X)\n",
    "    inertia_errors.append(model.inertia_)\n",
    "    silhouette_scores.append(silhouette_score(X, model.labels_))\n",
    "print(\"Inertia:\", inertia_errors)\n",
    "print()\n",
    "print(\"Silhouette Scores:\", silhouette_scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e533481-06df-48a1-ab7a-1d590fd73e69",
   "metadata": {
    "tags": []
   },
   "source": [
    "Now that we have both performance metrics for several different settings of `n_clusters`, let's make some line plots to see the relationship between the number of clusters in a model and its inertia and silhouette scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "23eb146d-54aa-4338-9f1e-70be64241831",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'K-Means Model: Inertia vs Number of Clusters')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot `inertia_errors` by `n_clusters`\n",
    "plt.plot(n_clusters, inertia_errors)\n",
    "plt.xlabel(\"Number of Clusters (k)\")\n",
    "plt.ylabel(\"Inertia\")\n",
    "plt.title(\"K-Means Model: Inertia vs Number of Clusters\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "63c717f9-fe1f-447b-814c-c3102a6a6712",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'K-Means Model: Silhouette Score vs Number of Clusters')"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot `silhouette_scores` vs `n_clusters`\n",
    "plt.plot(n_clusters, silhouette_scores)\n",
    "plt.xlabel(\"Number of Clusters\")\n",
    "plt.ylabel(\"Silhouette Scores\")\n",
    "plt.title(\"K-Means Model: Silhouette Score vs Number of Clusters\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "8f731ce5-87d5-4c31-806f-2d5184c6c741",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KMeans(n_clusters=4, random_state=42)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Build model\n",
    "final_model = KMeans(n_clusters=4, random_state=42)\n",
    "# Fit model to data4\n",
    "final_model.fit(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5cb0f75-4972-40f5-88c5-568769063827",
   "metadata": {
    "tags": []
   },
   "source": [
    "(In case you're wondering, we don't need an *Evaluate* section in this notebook because we don't have any test data to evaluate our model with.)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63474b1b-2841-43eb-b11c-da16b078e452",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Communicate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "670ddaf1-a5d3-4ea0-8486-bfbaf11011d0",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot \"HOUSES\" vs \"DEBT\" with final_model labels\n",
    "sns.scatterplot(\n",
    "    x=df[\"DEBT\"]/1e6,\n",
    "    y=df[\"HOUSES\"]/1e6,\n",
    "    hue=final_model.labels_,\n",
    "    palette=\"deep\"\n",
    ")\n",
    "plt.xlabel(\"Household Debt [$1M]\")\n",
    "plt.ylabel(\"Home Value [$1M]\")\n",
    "plt.title(\"Credit Fearful: Home Value vs. Household Debt\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "86c83bd0-4546-4457-96d3-f5a5e21bea9c",
   "metadata": {
    "deletable": false
   },
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
       "      <th>DEBT</th>\n",
       "      <th>HOUSES</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.488629e+04</td>\n",
       "      <td>1.031872e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.838410e+07</td>\n",
       "      <td>3.448400e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.472800e+06</td>\n",
       "      <td>1.407400e+07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.420929e+06</td>\n",
       "      <td>4.551429e+06</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           DEBT        HOUSES\n",
       "0  8.488629e+04  1.031872e+05\n",
       "1  1.838410e+07  3.448400e+07\n",
       "2  5.472800e+06  1.407400e+07\n",
       "3  2.420929e+06  4.551429e+06"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb = X.groupby(final_model.labels_).mean()\n",
    "xgb"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "101e4725-a0c9-4430-aba8-cc478819951a",
   "metadata": {
    "tags": []
   },
   "source": [
    "Before you move to the next task, print out the `cluster_centers_` for your `final_model`. Do you see any similarities between them and the DataFrame you just made? Why do you think that is?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "2c1279e4-83db-4f5a-a56b-6ba2271c900e",
   "metadata": {
    "deletable": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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T/M6zj2EWkmX985hnquqP0/L3ZCLflvV+S36l78qW6ZHjx7Mnz8fgGXLlnHCCSewatUqrrgi3RX/8Y9/nOnTp7c63d13383FF1/M/fffv2nYySefzIQJE5g4ceIW09ZSnncNPQ7sX2L4pIL315Ce3GNm1insvPPOTJ48mdGjR3P55ZdXPN3q1avZcccd8wusAzpdFxNmZvW2xx57sHHjRpYtWwbArFmzGDFixKbxt956K3vuuSdvvvkmI0aMYN26dSxdupT77mvrI9Nrw4nAzKwdCluxK2ka+sMf/sBJJ53EggUL3nMXv93XkJlZGz3//PN06dKFnXfeueJpxo4dy4oVK3gv3gLvRGBm1gbLly/njDPO4JxzzmnTmf3TTz/Nhg0b6NOnT47RtY+bhszMymhu63/nnXfo2rUrJ554Iueff/6m8cXXCC655BImTpy4aTpITUk33HADXbp0qXH05TkRmFmnUsntntW2YcOGFseNGzeOVatK34rc2nQAU6ZM6UhYVeOmITOzBudEYGbW4JwIzMwanBOBmVmDcyIwM2twTgRmZg3OicDMrIyePXtu9nnKlCmcc845mz5PnjyZvffem7333psxY8Ywe/bsTeOGDBnCihUrNn2eOXMmEyakx7S//PLLTJgwgeHDhzNs2DCOOuooABYtWkSPHj0269p66tSpAFx//fV85CMfYb/99mPffffl9ttv7/D6+XcEZta5VLtL8g52Rz59+nR+/OMfM3v2bPr27cu8efM45phjeOSRR9h1111bnfbSSy/l8MMP59xzzwXg8ccf3zRuzz333NRPUbOmpiauvPJK5s2bR+/evVm7dm1VuqxwjcDMrAO+853v8N3vfpe+fdPDFUeOHMmXvvQlrr322rLTLl26lIEDB276vN9++7VaftmyZfTq1WtTDaVnz55tejZxS5wIzMzKaO4qovl16aWXbhq3cOFCDjjggM3Kjxo1ioULF5ad79lnn82Xv/xlDjnkEK688kqWLFmyadxzzz232TJnzZrF8OHD2WWXXRg6dCinnHIKv/nNb6qyfm4aMjMro7A7aUjXCObMmdNi+YjY1CFdqY7pmocdccQRPP/889x1113ceeed7L///ixYsAAo3TQEcNddd/Hoo49y7733ct555zF37tw2PSCnFNcIzMw6YNiwYcydO3ezYfPmzWPYsGEA9OnT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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create side-by-side bar chart of `xgb`\n",
    "(xgb).plot(kind=\"bar\")\n",
    "plt.xlabel(\"Cluster\")\n",
    "plt.ylabel(\"Value [$1 million]\")\n",
    "plt.title(\"Mean Home Value & Household Debt by Cluster\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "51d8dea4-9330-49bb-aec5-7a6a91d4b316",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Proportion of Debt to Home value'}, xlabel='Cluster', ylabel='Proportion, Debt/Home'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "(xgb[\"DEBT\"]/xgb[\"HOUSES\"]).plot(\n",
    "    kind=\"bar\",\n",
    "    xlabel=\"Cluster\",\n",
    "    ylabel=\"Proportion, Debt/Home\",\n",
    "    title=\"Proportion of Debt to Home value\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7ff343c-2a60-4954-bec3-98983791df85",
   "metadata": {
    "tags": []
   },
   "source": [
    "In this plot, we have our four clusters spread across the x-axis, and the dollar amounts for home value and household debt on the y-axis. \n",
    "\n",
    "The first thing to look at in this chart is the different mean home values for the five clusters. Clusters 0 represents households with small to moderate home values, clusters 2 and 3 have high home values, and cluster 1 has extremely high values. \n",
    "\n",
    "The second thing to look at is the proportion of debt to home value. In clusters 1 and 3, this proportion is around 0.5. This suggests that these groups have a moderate amount of untapped equity in their homes. But for group 0, it's almost 1, which suggests that the largest source of household debt is their mortgage. Group 2 is unique in that they have the smallest proportion of debt to home value, around 0.4.\n",
    "\n",
    "This information could be useful to financial institution that want to target customers with products that would appeal to them. For instance, households in group 0 might be interested in refinancing their mortgage to lower their interest rate. Group 2 households could be interested in a [home equity line of credit](https://www.investopedia.com/home-equity-line-of-credit-heloc-definition-5217473) because they have more equity in their homes. And the bankers, Bill Gates, and Beyoncés in group 1 might want white-glove personalized wealth management. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db95983a-e4a4-49ba-9fdf-6a9d485f78c4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2a6e116-9b5d-400e-b10f-0914339cca6e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c922f829-71c0-4ebd-9e28-fda45664843d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffbcfa72-72d0-442a-9ea5-65bbe07de8f2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e108f6b-c54a-4696-8103-a73160f78ea7",
   "metadata": {},
   "outputs": [],
   "source": [
    "ddd"
   ]
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
