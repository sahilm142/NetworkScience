####---- INSTALL AND LOAD RESPECTIVE LIB ----####
#library("rJava")
#library("NLP")
#library("openNLP")
#library("RWeka")
#library("magrittr")
#library("twitteR")
library("tm")
library("rvest")
#library("tidytext")
#library("ggplot2")

p <-readLines("causal.txt")
myCorpus <- Corpus(VectorSource(p))
# convert to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))
# remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, removeURL)
# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
myCorpus <- tm_map(myCorpus,removeNumPunct)
# remove stopwords
#myStopwords <- c(setdiff(stopwords('english'), c("r", "big")),
#                "use", "see", "used", "via", "amp")
myCorpus <- tm_map(myCorpus, removeWords, stopwords("en"))
# remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)
myCorpus <- tm_map(myCorpus, content_transformer(removePunctuation))
removeQuoteS <- function(x) gsub("'"," ",x)
myCorpus <- tm_map(myCorpus,removeQuoteS)

tdm <- TermDocumentMatrix(myCorpus, control = list(wordLengths = c(1,Inf)))

#wordCloud

m <- as.matrix(tdm)
word.freq <- sort(rowSums(m),decreasing = T)
library(wordcloud)
pal <- brewer.pal(9, "BuGn")[-(1:4)]
wordcloud(words = names(word.freq), freq = word.freq,min.freq = 5,
          colors = pal)

####---- NAMED ENTITY RELATIONSHIP ----####
#lib tidyverse would override annotate fn from nlp
#library("rJava")
#library("RWeka")
#library("magrittr")
library("NLP");
library("tm");
library("openNLP");
library("openNLPmodels.en");
bio <- readLines("causal.txt")
#print(bio)
bio <- as.String(bio)
word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()
bio_annotations <- annotate(bio, list(sent_ann, word_ann))
class(bio_annotations)
head(bio_annotations)
bio_doc <- AnnotatedPlainTextDocument(bio, bio_annotations)
#sents(bio_doc) %>% head(2)
#words(bio_doc) %>% head(10)

#NER
person_ann <- Maxent_Entity_Annotator(kind = "person")
location_ann <- Maxent_Entity_Annotator(kind = "location")
organization_ann <- Maxent_Entity_Annotator(kind = "organization")
#money_ann <- Maxent_Entity_Annotator(kind = "money")

pipeline <- list(sent_ann, word_ann, person_ann, location_ann,
                 organization_ann )
bio_annotations <- annotate(bio, pipeline)
bio_doc <- AnnotatedPlainTextDocument(bio, bio_annotations)

# Extract entities from an AnnotatedPlainTextDocument
entities <- function(doc, kind) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
  s <- doc$content
  a <- annotations(doc)[[1]]
  if(hasArg(kind)) {
    k <- sapply(a$features, `[[`, "kind")
    s[a[k == kind]]
  } else {
    s[a[a$type == "entity"]]
  }
}
entities(bio_doc)
person_ner<-entities(bio_doc,kind = "person")
organization_ner <- entities(bio_doc, kind = "organization")
location_ner <- entities(bio_doc, kind = "location")



####---- PART OF SPEECH TAGGING ----####
#CC= Coordinating conjunction 
#CD= Cardinal number 
#DT= Determiner 
#EX= Existential there 
#FW= Foreign word 
#IN= Preposition or subordinating conjunction 
#JJ= Adjective JJR= Adjective, comparative 
#JJS= Adjective, superlative 
#LS= List item marker 
#MD= Modal 
#NN= Noun, singular or mass
#NNS= Noun, plural 
#NNP= Proper noun, singular 
#NNPS= Proper noun, plural 
#PDT= Predeterminer 
#POS= Possessive ending 
#PRP= Personal pronoun 
#PRP$= Possessive pronoun 
#RB= Adverb 
#RBR= Adverb, comparative 
#RBS= Adverb, superlative 
#RP= Particle 
#SYM= Symbol 
#TO= to 
#UH= Interjection 
#VB= Verb, base form 
#VBD= Verb, past tense 
#VBG= Verb, gerund or present participle 
#VBN= Verb, past participle 
#VBP= Verb, non¬3rd person singular present 
#VBZ= Verb, 3rd person singular present 
#WDT= Wh¬determiner 
#WP= Wh¬pronoun 
#WP$= Possessive wh¬pronoun 
#WRB= Wh¬adverb



sent_token_annotator <-Maxent_Sent_Token_Annotator()
word_token_annotator <-Maxent_Word_Token_Annotator()
myCorpus_str <- as.String(myCorpus)
a2 <-annotate(myCorpus_str, list(sent_token_annotator, word_token_annotator))
pos_tag_annotator <-Maxent_POS_Tag_Annotator()
pos_tag_annotator
#>An annotator inheriting from classes
#>  Simple_POS_Tag_Annotator Annotator
#>with description
#>  Computes POS tag annotations using the Apache OpenNLP Maxent Part of
#>  Speech tagger employing the default model for language 'en'
a3 <-annotate(myCorpus_str, pos_tag_annotator, a2)
a3
## Variant with POS tag probabilities 
head(annotate(myCorpus_str, Maxent_POS_Tag_Annotator(probs = TRUE), a2))

## Determine the distribution of POS tags for word tokens.
a3_dist <-subset(a3, type == "word")
tags <-sapply(a3_dist$features, '[[', "POS")
table(tags)   # Total numbers of each tag

#Extract token/POS pairs 
sprintf("%s/%s",myCorpus_str[a3_dist],tags)


########----SENTIMENT ANALYSIS----#######
library("rJava")
library("NLP")
library("openNLP")
library("RWeka")
library("magrittr")
library("tidytext")
library("ggplot2")
library(tidyverse)
#p_line <- p %>% 
# dplyr::mutate(h_number = row_number())

p_df <- data_frame(text = p)
p_tidy <- p_df %>% unnest_tokens(word, text)                   #tokenization
p_tidy <- p_tidy %>% anti_join(stop_words)
set.seed(0)

#Sentiment afinn
sentiment_afinn <- p_tidy %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(row_number()) %>% 
  summarise(score_afinn = sum(score)) %>%
  ungroup()

#Sentiment bing
sentiment_bing <- p_tidy %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(row_number(), sentiment) %>%
  spread(sentiment, n, fill=0) %>%
  mutate(score_bing = positive - negative) %>%
  select(-positive, -negative) %>%
  ungroup()

#Sentiment nrc
sentiment_nrc <- p_tidy %>% 
  inner_join(get_sentiments("nrc")) %>% 
  count(row_number(), sentiment) %>%
  spread(sentiment, n, fill=0) %>%
  setNames(c(names(.)[1],paste0('nrc_', names(.)[-1]))) %>%
  mutate(score_nrc = nrc_positive - nrc_negative) %>%
  ungroup()

#Combine all sentiment rating and fill the missing with 0
p_sentiments <- Reduce(full_join,
                       list(sentiment_nrc, sentiment_bing, sentiment_afinn)) %>% 
  mutate_each(funs(replace(., which(is.na(.)), 0)))

#See all sentiment 
p_cors <- p_sentiments %>%
  select(starts_with("score")) %>%
  cor() %>%
  round(digits=2)

upper<-p_cors
upper[upper.tri(p_cors)]<-""
knitr::kable(upper, format="html", booktabs = TRUE)
upper


p_sentiments %>%
  gather(emotion, intensity,starts_with("nrc_")) %>%
  filter(intensity > 0) %>%
  mutate(emotion = substring(emotion,5)) %>%
  ggplot(aes(x = score_nrc, y = score_bing)) +
  geom_hex(bins=5) +
  facet_wrap(~emotion, nrow = 2)

library(scales)

senti_afinn <- p_tidy %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = row_number()) %>% 
  summarise(sentiment = sum(score)) %>%
  mutate(method = "AFINN")

bing_and_nrc <- bind_rows(p_tidy %>% 
                            inner_join(get_sentiments("bing"))%>%
                            mutate(method = "Bing et al."),
                          p_tidy %>%
                            inner_join(get_sentiments("nrc")%>%
                                         filter(sentiment %in% c("positive",
                                                                 "negative")))%>%
                            mutate(method = "NRC")) %>%
  count(method, index = row_number(),sentiment)%>%
  spread(sentiment, n, fill=0)%>%
  mutate(sentiment = positive - negative)
bind_rows(senti_afinn, bing_and_nrc)%>% 
  ggplot(aes(index, sentiment, fill = method))+
  geom_col(show.legend = FALSE)+
  facet_wrap(~method, ncol = 1, scales = "free_y")
#bing sentiment plot
bing_word_counts <- p_tidy %>% inner_join(get_sentiments("bing"))%>%
  count(word, sentiment, sort = TRUE)%>%
  ungroup()
bing_word_counts

bing_word_counts %>% 
  group_by(sentiment) %>%
  top_n(15) %>%
  ungroup() %>%
  mutate(word = reorder(word, n))%>%
  ggplot(aes(word, n, fill = sentiment))+
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment", x = NULL)+
  coord_flip()
#define custom stop words to remove causal words
custom_stop_words <- bind_rows(data_frame(word = c("caused",
                                                   "causing","causing"),
                                          lexicon = c("custom")),
                               stop_words)
p_tidy %>%
  anti_join(custom_stop_words) %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))




####---- TOPIC MODELLING ----####
library(magrittr) # for %$% operator 
library(tidyverse) 
library(tidytext)  # for easy handling of text
library(ldatuning) # for mathematical hint of number of topics
library(topicmodels) 


myCorpus_dtm <- DocumentTermMatrix(myCorpus, 
                                   control = list(stemming = FALSE,removeNumbers = TRUE,
                                                  removePunctuation =TRUE,
                                                  wordLengths = c(3,Inf)))
#findFreqTerms(myCorpus_dtm,3)

k <- 10
#myCorpus_dtm <- removeSparseTerms(myCorpus_dtm,0.99)

control_LDA_Gibbs <- list(alpha = 50/k, estimate.beta = T, 
                          verbose = 0, prefix = tempfile(), 
                          save = 0, 
                          keep = 50, 
                          seed = 980, 
                          nstart = 1, best = T,
                          delta = 0.1,
                          iter = 2000, 
                          burnin = 100, 
                          thin = 2000) 
rowTotals <- apply(myCorpus_dtm , 1, sum) #Find the sum of words in each Document
myCorpus_dtm.new   <- myCorpus_dtm[rowTotals> 0, ]
topic_model <- LDA(myCorpus_dtm.new, k, method = "Gibbs", 
                   control = control_LDA_Gibbs)

terms(topic_model, 15)
topic_assignments_by_docs <- topics(topic_model)




#library(parallel)
#many_models <- mclapply(seq(2, 35, by = 1), function(x) 
#  {LDA(myCorpus_dtm.new, x, method = "Gibbs", control = control_LDA_Gibbs)} )
#many_models.logLik <-
#  as.data.frame(as.matrix(lapply(many_models, logLik)))
#plot(2:35, unlist(lda.models.gibbs.logLik), 
#     xlab="Number of Topics", ylab="Log-Likelihood")
##Topic assignment of all the documents


