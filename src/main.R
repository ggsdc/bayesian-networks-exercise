library(bnlearn)
library(rio)
library(dplyr)
library(BiocManager)
library(Rgraphviz)


data <- import(file="https://weka.8497.n7.nabble.com/file/n23121/credit_fruad.arff")

data <- data %>% 
    mutate_all(as.factor)

net <- hc(data)
graphviz.plot(net, layout="fdp", shape="rectangle")

# str.diff = boot.strength(data, R = 200, algorithm = "hc")
# head(str.diff)
# avg.net = averaged.network(str.diff)
# graphviz.plot(avg.net, layout="fdp", shape="rectangle")

fitted.net <- bn.fit(net, data)


xval = bn.cv(data, bn = "pc.stable",loss="pred-lw-cg", loss.args = list(target = "class"), method = "k-fold", k=10)
xval
