library(bnlearn)
library(rio)
library(dplyr)
library(BiocManager)
library(Rgraphviz)
library(ggplot2)
library(tidyr)
library(purrr)


data <- import(file="https://weka.8497.n7.nabble.com/file/n23121/credit_fruad.arff")

t <- data %>%
    select(-current_balance, -cc_age, -credit_usage) %>% 
    gather(key="key",value="value", over_draft:foreign_worker)


t %>%   
    ggplot() + 
    geom_histogram(aes(value, group=class, fill=class), stat="count", position="dodge") + 
    facet_wrap(~ key, scales = "free") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1),
          text = element_text(size=15))

t2 <- data %>%
    select(current_balance, cc_age, credit_usage, class) %>% 
    gather(key="key",value="value", current_balance:credit_usage)

t2 %>%   
    ggplot() + 
    geom_density(aes(value, group=class, color=class, fill=class), alpha=0.1) + 
    facet_wrap(~ key, scales = "free") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1),
          text = element_text(size=15))

net <- hc(data)
graphviz.plot(net, layout="fdp", shape="rectangle")

set.seed(666)

data.train <- data %>% 
    sample_frac(0.8)

data.test <- data %>% 
    anti_join(data.train)

xval.hc = bn.cv(data.train, bn = "hc",loss="pred-lw-cg", loss.args = list(target = "class"), method = "k-fold", k=8)

xval.tabu = bn.cv(data.train, bn = "tabu",loss="pred-lw-cg", loss.args = list(target = "class"), method = "k-fold", k=8)

xval.hc
xval.tabu

perf.hc <- data.frame()
perf.tabu <- data.frame()

for (i in 1:length(xval.hc)){
    aux <- data.frame(obs = xval.hc[[i]]$observed, pred=xval.hc[[i]]$predicted)
    perf.hc <- bind_rows(perf.hc, aux)
    aux <- data.frame(obs = xval.tabu[[i]]$observed, pred=xval.tabu[[i]]$predicted)
    perf.tabu <- bind_rows(perf.tabu, aux)
}

perf.hc %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())

perf.tabu %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())

net <- hc(data.train)
graphviz.plot(net, layout="fdp", shape="rectangle")

net.train <- bn.fit(net, data.train)
