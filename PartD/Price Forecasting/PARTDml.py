from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors


ethereum = "hdfs://andromeda.eecs.qmul.ac.uk"
path = ethereum + "/user/rsk01/ethereum.csv"


sc = SparkContext()
sqlcontext = SQLContext(sc)

df1 = sqlcontext.read.format('csv') \
					 .options(header='true', inferschema='true') \
					 .load(path)

assembler = VectorAssembler(inputCols = ['SplyCur', 'TxTfrCnt'], outputCol = 'features')

output = assembler.transform(df1)

finaldata = output.select('features', 'PriceUSD').filter(df1['PriceUSD'].isNotNull())

traindata, testdata = finaldata.randomSplit([0.7,0.3])

linearregression = LinearRegression(labelCol='PriceUSD', maxIter=100, regParam=0.3, elasticNetParam=0.5)

linearmodel = linearregression.fit(traindata)

testresult = linearmodel.evaluate(testdata)

print(testresult.rootMeanSquaredError)

print(testresult.r2)

data1 = testdata.select('features')

predictions = linearmodel.transform(data1)


predictions.show()
