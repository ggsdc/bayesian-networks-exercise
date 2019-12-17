library(bnlearn)
library(rio)
library(dplyr)
# library(BiocManager)
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

set.seed(9)

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



## Constraint based

xval.pc <- bn.cv(data.train, bn = "pc.stable",loss="pred-lw-cg", loss.args = list(target = "class"), method = "k-fold", k=8)
xval.gs <- bn.cv(data.train %>% mutate_all(as.factor), bn = "gs",loss="pred", loss.args = list(target = "class"), method = "k-fold", k=8)
xval.iamb <- bn.cv(data.train %>% mutate_all(as.factor), bn = "iamb",loss="pred", loss.args = list(target = "class"), method = "k-fold", k=8)
xval.fiamb <- bn.cv(data.train %>% mutate_all(as.factor), bn = "fast.iamb",loss="pred", loss.args = list(target = "class"), method = "k-fold", k=8)


xval.pc
xval.gs
xval.iamb
xval.fiamb

perf.pc <- data.frame()
perf.gs <- data.frame()
perf.iamb <- data.frame()
perf.fiamb <- data.frame()

for (i in 1:length(xval.pc)){
    aux <- data.frame(obs = xval.pc[[i]]$observed, pred=xval.pc[[i]]$predicted)
    perf.pc <- bind_rows(perf.pc, aux)
    
    aux <- data.frame(obs = xval.gs[[i]]$observed, pred=xval.gs[[i]]$predicted)
    perf.gs <- bind_rows(perf.gs, aux)
    
    aux <- data.frame(obs = xval.iamb[[i]]$observed, pred=xval.iamb[[i]]$predicted)
    perf.iamb <- bind_rows(perf.iamb, aux)
    
    aux <- data.frame(obs = xval.fiamb[[i]]$observed, pred=xval.fiamb[[i]]$predicted)
    perf.fiamb <- bind_rows(perf.fiamb, aux)
}

perf.pc %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())

perf.gs %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())

perf.iamb %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())

perf.fiamb %>% 
    group_by(obs, pred) %>% 
    summarise(count=n())
