{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.externals import joblib\n",
    "import xgboost as xgb\n",
    "from xgboost.sklearn import XGBClassifier\n",
    "from sklearn.cross_validation import StratifiedKFold\n",
    "from sklearn import cross_validation\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import time    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_data = pd.read_csv('train_data_40.csv')\n",
    "y = train_data['label']\n",
    "x = train_data.drop(['label'],axis=1,inplace = False)\n",
    "\n",
    "#按照比例划分数据集\n",
    "X_train, X_test, y_train, y_test = cross_validation.train_test_split( x, y, stratify=y, test_size=0.3, random_state=1)\n",
    "X_train = X_train.reset_index(drop=True)\n",
    "y_train = y_train.reset_index(drop=True)\n",
    "X_test = X_test.reset_index(drop=True)\n",
    "y_test = y_test.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_data = pd.read_csv('test_data_final_40.csv')\n",
    "temp_time = test_data['time']\n",
    "temp_group = test_data['group']\n",
    "X_test = test_data.drop(['time','group'],axis=1,inplace = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "pre_label = predict(clff,X_test,0.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "can't set attribute",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__setattr__\u001b[0;34m(self, name, value)\u001b[0m\n\u001b[1;32m   2702\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2703\u001b[0;31m                     \u001b[0mobject\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__setattr__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2704\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mAttributeError\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: can't set attribute",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-29-61a0726639d0>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtemp_time\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'int'\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32mD:\\anaconda\\soft\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__setattr__\u001b[0;34m(self, name, value)\u001b[0m\n\u001b[1;32m   2703\u001b[0m                     \u001b[0mobject\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__setattr__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2704\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mAttributeError\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2705\u001b[0;31m                 \u001b[0mobject\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__setattr__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2706\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m   2707\u001b[0m     \u001b[1;31m# ----------------------------------------------------------------------\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: can't set attribute"
     ]
    }
   ],
   "source": [
    "new_time = (lambda x: int(x))(x) for x in "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def output(final_label):\n",
    "    start_time =[]\n",
    "    end_time = []\n",
    "    for i in range(len(final_label)-1):\n",
    "        if (final_label[i] == 0) & (final_label[i+1] == 1):\n",
    "            start_time.append(temp_time[i+1])\n",
    "        if (final_label[i] == 1) & (final_label[i+1] == 0):\n",
    "            end_time.append(temp_time[i+1]-1)\n",
    "    final_pre_result = pd.DataFrame(start_time,columns = ['startTime'])\n",
    "    final_pre_result['endTime'] = end_time\n",
    "    final_pre_result.to_csv('test_08_results.csv',index = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "n_estimators = 0.6, score = 98.65822129541641, Negtive =305(290)\n",
      "\n",
      "n_estimators = 0.65, score = 98.71266380412722, Negtive =300(290)\n",
      "\n",
      "n_estimators = 0.7, score = 98.5552021375192, Negtive =298(289)\n",
      "\n",
      "n_estimators = 0.75, score = 98.58786764274569, Negtive =295(289)\n",
      "\n",
      "n_estimators = 0.8, score = 98.44129447787985, Negtive =292(288)\n",
      "\n",
      "n_estimators = 0.85, score = 97.9471324745715, Negtive =288(285)\n",
      "\n",
      "used time :19.952706813812256\n"
     ]
    }
   ],
   "source": [
    "t1 = time.time()  \n",
    "#parameter = range(30,35,5)\n",
    "parameter =[0.6,0.65,0.7,0.75,0.8,0.85]\n",
    "for i in range(len(parameter)):\n",
    "    #clff = LR_model(X_train,list(y_train))\n",
    "    #clff = LR_model(X_train,list(y_train),parameter[i])\n",
    "    clff = xgb_model(X_train,y_train,1)\n",
    "    #clff = RF_model(X_train,list(y_train),parameter[i])\n",
    "\n",
    "    s,n,TP = cal_score(clff,X_test,y_test,parameter[i])\n",
    "    print( \"n_estimators = \"+ str(parameter[i]) + \", score = \" + str(s) + \", Negtive =\" + str(n) + \n",
    "          \"(\" + str(TP) + \")\" + \"\\n\")\n",
    "\n",
    "\n",
    "t2 = time.time()\n",
    "print(\"used time :\" + str(t2-t1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def xgb_model(train_data,y_train_label,paramter):\n",
    "    weight_list = []\n",
    "    for j in range(len(y_train_label)):\n",
    "        if y_train_label[j] == 1:\n",
    "            weight_list.append(20)\n",
    "        if y_train_label[j] == 0:\n",
    "            weight_list.append(1)     \n",
    "    clf = XGBClassifier(\n",
    "    silent=0 ,\n",
    "    learning_rate= 0.3, # 如同学习率\n",
    "    min_child_weight=1, \n",
    "    max_depth = 5, # 构建树的深度，越大越容易过拟合\n",
    "    gamma = 0.2,  \n",
    "    subsample=1, # 随机采样训练样本 训练实例的子采样比\n",
    "    max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。\n",
    "    colsample_bytree=1, # 生成树时进行的列采样 \n",
    "    reg_lambda = 1.2,  # 控制模型复杂度的权重值的L2正则化项参数\n",
    "    reg_alpha = 0.7, # L1 正则项参数\n",
    "    scale_pos_weight = 2.2, \n",
    "    n_estimators = 65, #树的个数\n",
    "    seed=0 #随机种子\n",
    "    )\n",
    "    clf.fit(train_data,y_train_label,weight_list)\n",
    "    return clf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def cal_score(cl,test_data,y_test_label,threshold):\n",
    "    #pre_label = cl.predict(test_data)\n",
    "    pre_label = [0]*len(y_test_label)\n",
    "    pre_label_pro = cl.predict_proba(test_data)\n",
    "\t\n",
    "    for i in range(len(pre_label_pro)):\n",
    "        if pre_label_pro[i][1] > threshold:\n",
    "            pre_label[i] = 1   \n",
    "\t\t\t\n",
    "    n_f = 0\n",
    "    n_n = 0\n",
    "    FP = 0\n",
    "    FN = 0\n",
    "    TP = 0\n",
    "    n_label = 0\n",
    "    for i in range(0,len(y_test_label)):\n",
    "        if pre_label[i] == 1:\n",
    "            n_label = n_label+1\n",
    "        if y_test_label[i] == 1:\n",
    "            n_f = n_f + 1\n",
    "            if pre_label[i] == 0:\n",
    "                FP = FP + 1\n",
    "            else:\n",
    "                TP = TP + 1\n",
    "        else: #y_test_label[i] == 0\n",
    "            n_n = n_n + 1\n",
    "            if pre_label[i] == 1:\n",
    "                FN = FN + 1\n",
    "    score = (1-0.5*(FN/n_n)-0.5*(FP/n_f))*100\n",
    "    return score,n_label,TP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def predict(cl,test_data,threshold):\n",
    "    pre_label = [0]*len(test_data)\n",
    "    pre_label_pro = cl.predict_proba(test_data)\n",
    "    for i in range(len(pre_label_pro)):\n",
    "        if pre_label_pro[i][1] > threshold:\n",
    "            pre_label[i] = 1 \n",
    "    return pre_label"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [default]",
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
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
