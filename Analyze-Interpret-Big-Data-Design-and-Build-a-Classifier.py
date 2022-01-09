#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.mllib.stat import Statistics


# In[48]:


from pyspark.sql import SparkSession


# In[49]:


spark = SparkSession.builder.appName('attacks-app').getOrCreate()


# In[50]:


data_df = spark.read.csv('/opt/UNSW-NB15.csv', header = True, inferSchema = True).toDF(
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','attack_label')


# In[51]:


data_df.printSchema()


# In[52]:


import pandas as pd
pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[53]:


import pandas as pd
pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[54]:


# selected varables for the Descriptive statistics (level of attack)
num_cols = ['sbytes', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_dst', 'dur']
data_df.select(num_cols).describe().show()


# In[55]:


from pyspark.sql.functions import col, skewness, kurtosis
var = 'is_ftp_login'
data_df.select(skewness(var),kurtosis(var)).show()


# In[56]:


import seaborn as sns
x = data_df.select(var).toPandas()

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 2, 1)
ax = sns.boxplot(data=x)

ax = fig.add_subplot(1, 2, 2)
ax = sns.violinplot(data=x)


# In[57]:


from pyspark.mllib.stat import Statistics
import pandas as pd

corr_data = data_df.select(num_cols)

col_names = corr_data.columns
features = corr_data.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

print(corr_df.to_string())


# In[58]:


data_df.stat.crosstab("is_ftp_login", "ct_ftp_cmd").show()


# In[59]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
assembler = VectorAssembler(inputCols = ['is_ftp_login', 'ct_ftp_cmd','dwin', 'dmeansz', 'dur'], outputCol = "features")
assembled = assembler.transform(data_df)
pearson_corr = Correlation.corr(assembled, "features")
corr_list = pearson_corr.head()[0].toArray().tolist()
pearson_corr_df = spark.createDataFrame(corr_list)
pearson_corr_df.show(truncate=False)


# In[60]:


data_df = data_df.drop("srcip", "sport", "dstip", "dsport")


# In[61]:


numericfeatures = [t[0] for t in data_df.dtypes if t[1] == 'int' or t[1] == 'long'or t[1] == 'double']


# In[62]:


data_df.select(numericfeatures).describe()


# In[63]:


data_df.select(numericfeatures).describe().toPandas().transpose()


# In[64]:


import pandas as pd
pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[65]:


import pandas as pd
pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[66]:


data_df = data_df.select('proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat', 'attack_label')
cols = data_df.columns
data_df.printSchema()


# In[67]:


from pyspark.ml.feature import OneHotEncoder


# In[68]:


from pyspark.ml.feature import StringIndexer


# In[69]:


from pyspark.ml.feature import VectorAssembler


# In[70]:


categoricalColumns = ['proto', 'service', 'attack_cat']
stages = []


# In[71]:


for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]


# In[72]:


label_stringIdx = StringIndexer(inputCol = 'attack_label', outputCol = 'label')
stages += [label_stringIdx]


# In[73]:


numericCols = ['dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']


# In[74]:


assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# In[75]:


data_df.printSchema()


# In[76]:


data_df = data_df.na.fill('Normal')


# In[77]:


data_df.show()


# In[78]:


from pyspark.sql.functions import col,when
data_df=data_df.select([when(col(service)=="-",'Normal').otherwise(col(service)).alias(service) for service in data_df.columns])
data_df.show()


# In[79]:


data_df.head()


# In[80]:


import pandas as pd
pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[81]:


data_df = data_df.selectExpr("cast(proto as string) proto", "cast(state as string) state", "cast(dur as float) dur", "cast(sbytes as int) sbytes", "cast(dbytes as int) dbytes"
, "cast(sttl as int) sttl", "cast(dttl as int) dttl", "cast(sloss as int) sloss", "cast(dloss as int) dloss", "cast(service as string) service"
, "cast(Sload as float) Sload", "cast(Dload as float) Dload", "cast(Spkts as int) Spkts", "cast(Dpkts as int) Dpkts", "cast(swin as int) swin"
, "cast(dwin as int) dwin", "cast(stcpb as bigint) stcpb", "cast(dtcpb as bigint) dtcpb", "cast(smeansz as int) smeansz", "cast(dmeansz as int) dmeansz"
, "cast(trans_depth as int) trans_depth", "cast(res_bdy_len as int) res_bdy_len", "cast(Sjit as float) Sjit", "cast(Djit as float) Djit", "cast(Stime as bigint) Stime"
, "cast(Ltime as bigint) Ltime", "cast(Sintpkt as float) Sintpkt", "cast(Dintpkt as float) Dintpkt", "cast(tcprtt as float) tcprtt"
, "cast(synack as float) synack"
, "cast(ackdat as float) ackdat"
, "cast(is_sm_ips_ports as int) is_sm_ips_ports"
, "cast(ct_state_ttl as int) ct_state_ttl"
, "cast(ct_flw_http_mthd as int) ct_flw_http_mthd"
, "cast(is_ftp_login as int) is_ftp_login"
, "cast(ct_ftp_cmd as int) ct_ftp_cmd"
, "cast(ct_srv_src as int) ct_srv_src"
, "cast(ct_srv_dst as int) ct_srv_dst"
, "cast(ct_dst_ltm as int) ct_dst_ltm"
, "cast(ct_src_ltm as int) ct_src_ltm"
, "cast(ct_src_dport_ltm as int) ct_src_dport_ltm"
, "cast(ct_dst_sport_ltm as int) ct_dst_sport_ltm"
, "cast(ct_dst_src_ltm as int) ct_dst_src_ltm"
, "cast(attack_cat as string) attack_cat"
, "cast(attack_label as string) attack_label")
data_df.printSchema()
data_df.show(truncate=False)
data_df.printSchema()
data_df.show(truncate=False)


# In[82]:


data_df.printSchema()


# In[83]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(data_df)
data_df = pipelineModel.transform(data_df)
selectedCols = ['label', 'features'] + cols
data_df = data_df.select(selectedCols)
data_df.printSchema()


# In[84]:


import pandas as pd


# In[85]:


pd.DataFrame(data_df.take(5), columns=data_df.columns).transpose()


# In[86]:


train, test = data_df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[87]:


from pyspark.ml.classification import LogisticRegression


# In[88]:


lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train)


# In[89]:


import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[90]:


trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[91]:


pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[92]:


predictions = lrModel.transform(test)
predictions.select('attack_cat', 'dur', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[93]:


predictions = lrModel.transform(test)
predictions.select('attack_cat', 'dur', 'label', 'rawPrediction', 'prediction', 'probability','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm').show(10)


# In[94]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[95]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('attack_cat', 'dur', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[96]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[97]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('attack_cat', 'dur', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[98]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[99]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('attack_cat', 'dur', 'label', 'rawPrediction', 'prediction', 'probability').show(10)


# In[100]:


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


# In[101]:


print(gbt.explainParams())


# In[102]:


# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(train)


# In[103]:


# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))


# In[104]:


trainingSummary = lrModel.summary


# In[105]:


trainingSummary.accuracy


# In[106]:


objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)


# In[107]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[108]:


print("Multinomial coefficients: " + str(lrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(lrModel.interceptVector))


# In[109]:


import matplotlib.pyplot as plt
import numpy as np


# In[110]:


predictions = lrModel.transform(test)
predictions.select('dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm').show(10)


# In[111]:


predictions = lrModel.transform(test)
predictions.select('service', 'attack_cat', 'dur','sbytes','dbytes','dttl','sloss','dloss','Sload','Dload','trans_depth','res_bdy_len','Sjit').show(10)


# In[117]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.util import MLUtils
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[118]:


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Test Error = %g" % (1.0 - accuracy))


# In[115]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('label', 'attack_cat', 'dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm').show(10)


# In[119]:


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))


# In[120]:


print(rf.explainParams())


# In[123]:


from pyspark.sql.functions import regexp_replace, col
data = data.withColumn('attack_cat', regexp_replace(col("attack_cat"), " ", ""))
data.show(truncate =False)


# In[ ]:


from pyspark.sql.functions import regexp_replace, col
data = data.withColumn('service', regexp_replace(col("service"), " ", ""))
data.show(truncate =False)


# In[ ]:


from pyspark.sql.functions import col
data.groupBy("attack_cat").count().orderBy(col("count").desc()).show()


# In[ ]:


from pyspark.sql.functions import col
data.groupBy("service").count().orderBy(col("count").desc()).show()


# In[122]:


from pyspark.sql.functions import col
data.groupBy("attack_cat").count().orderBy(col("count").desc()).show()


# In[ ]:


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[ ]:


# Fit the model
lrModel = lr.fit(train)


# In[ ]:


# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))


# In[ ]:


trainingSummary = lrModel.summary


# In[ ]:


# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)


# In[ ]:


# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))


# In[ ]:


print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))


# In[ ]:


print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))


# In[ ]:


print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))


# In[ ]:


print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))


# In[ ]:


accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s" % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))


# In[ ]:


lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(train)


# In[ ]:


predictions = lrModel.transform(test)


# In[ ]:


predictions.filter(predictions['prediction'] == 0).select("service","attack_cat","probability","label","prediction").orderBy("probability", ascending=False).show(n = 20, truncate = 30)


# In[ ]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)


# In[ ]:


from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1)
model = nb.fit(train)
predictions = model.transform(test)


# In[ ]:


predictions.filter(predictions['prediction'] == 0).select("service","attack_cat","probability","label","prediction").orderBy("probability", ascending=False).show(n = 20, truncate = 30)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 100, maxDepth = 4, maxBins = 32)


# In[ ]:


# Train model with Training Data
rfModel = rf.fit(train)


# In[ ]:


predictions = rfModel.transform(test)
predictions.filter(predictions['prediction'] == 0).select("service","attack_cat","probability","label","prediction").orderBy("probability", ascending=False).show(n = 10, truncate = 30)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)


# In[ ]:


drop_list = ['proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','Sload','dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime',
    'Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm', 'Dload', 'attack_label']
	
data = data_df.select([column for column in data_df.columns if column not in drop_list])
data.show(5)


# In[ ]:


data.printSchema()


# In[ ]:


from pyspark.sql.functions import col
data.groupBy("attack_cat").count().orderBy(col("count").desc()).show()


# In[ ]:


from pyspark.sql import functions as F
data = data.withColumn("attack_cat", F.when(F.col("attack_cat")=='Reconnaissance ', "Reconnaissance").otherwise(F.col("attack_cat")))
data = data.withColumn("attack_cat", F.when(F.col("attack_cat")=='Fuzzers ', "Reconnaissance").otherwise(F.col("attack_cat")))
data = data.withColumn("attack_cat", F.when(F.col("attack_cat")=='Shellcode ', "Reconnaissance").otherwise(F.col("attack_cat")))


# In[ ]:


data.groupBy("attack_cat").count().orderBy(col("count").desc()).show()


# In[ ]:


data.groupBy("service").count().orderBy(col("count").desc()).show()


# In[ ]:


from pyspark.ml.feature import VectorAssembler
colmss = ['sbytes','dbytes','dur','sttl','dttl','sloss','dloss','Sload','Dload','Spkts','Dpkts','swin','dwin','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit',
          'Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat']


# In[ ]:


feature = VectorAssembler(inputCols= colmss[:],outputCol="feature")
sdf1 = feature.transform(data_df)


# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#from pyspark.mllib.util import MLUtils


# In[ ]:


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="attack_cat", outputCol="indexedLabel").fit(sdf1)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="feature", outputCol="indexedFeatures", maxCategories= 15).fit(sdf1)


# In[ ]:


# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = sdf1.randomSplit([0.7, 0.3])


# In[ ]:


# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")


# In[ ]:


# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])


# In[ ]:


# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)


# In[ ]:


# Make predictions.
predictions = model.transform(testData)


# In[ ]:


# Select example rows to display.
predictions.select("prediction", "indexedLabel", "feature").show(5)


# In[ ]:


# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


# In[ ]:


treeModel = model.stages[2]
print(treeModel) # summary only
print("Accuracy = ", accuracy)


# In[ ]:


tree = model.stages[-1]

display(tree) #visualize the decision tree model
print(tree.toDebugString) #print the nodes of the decision tree model


# In[ ]:


#Plot Between Actual and Predicted attck_cat
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

sbdf = predictions.select('prediction','indexedLabel').toPandas()


# In[ ]:


# PLOT 
ax = sbdf.plot(kind='scatter', figsize = (6,6), x='prediction', y='indexedLabel', color='blue', alpha = 1, label='Actual vs. predicted')
fit = np.polyfit(sbdf['prediction'], sbdf['indexedLabel'], deg=1)
ax.set_title('Actual vs. Predicted Attck Record')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.plot(sbdf['prediction'], fit[0] * sbdf['prediction'] + fit[1], color='magenta')
plt.axis([-1, 20, -1, 20])
plt.show(ax)


# In[ ]:


#Covarience on some pairs of columns

print("Covariance between sbytes and dbytes", data_df.stat.cov('sbytes', 'dbytes'))
print("Covariance between sloss and dloss", data_df.stat.cov('sloss', 'dloss'))
print("Covariance between Sload and Dload",data_df.stat.cov('Sload', 'Dload'))
print("Covariance between swin and dwin",data_df.stat.cov('swin', 'dwin'))


# In[ ]:


# correlation on some pairs of columns
print("Correlation between sbytes and dbytes", data_df.stat.corr('sbytes', 'dbytes'))
print("Correlation between sloss and dloss", data_df.stat.corr('sloss', 'dloss'))
print("Correlation between Sload and Dload",data_df.stat.corr('Sload', 'Dload'))
print("Correlation between swin and dwin",data_df.stat.corr('swin', 'dwin'))
print("Correlation between swin and dwin",data_df.stat.corr('sttl', 'dttl'))


# In[ ]:


data_df.stat.freqItems(['sbytes','dbytes','dur','sttl','dttl','sloss','dloss','Sload','Dload','Spkts','Dpkts','swin','dwin',
                    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit'], 0.4).collect()[0]


# In[ ]:


display(data_df)

