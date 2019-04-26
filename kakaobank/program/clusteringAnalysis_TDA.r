rm(list=ls())

library(TDAmapper)
library(ggplot2)
library(igraph)

# Load data
proj_path='/Users/skyeong/pythonwork/kakaobank/'

#for (year in c(2008:2015)) {
for (year in c(2008:2008)) {
    fn1 = sprintf('%s/results/user_activity_y%d.csv',proj_path,year)
  df1 = read.csv(fn1,header=T,row.names = 'user_id')
  fn2 = sprintf('%s/results/user_entropy_y%d.csv',proj_path,year)
  df2 = read.csv(fn2,header=T,row.names = 'user_id')
  
  # Concatenate two data.frame: activity + entropy
  df = data.frame(df1,df2)
  df = df[complete.cases(df), ]
  
  # remove extremely low activity users
  data = df[(df$a2q_src>=2) & (df$c2q_src>=2) & (df$c2a_src>=2),] 

  # To reduce data size, random sampling was applied.
  if (dim(data)[1]>10000) {
    set.seed(123458)
    data = data[sample(nrow(data), 10000), ]
  }
  
  # Compute Expert and Learner Scores
  raw_vars=c('a2q_src','a2q_tgt','c2q_src', 'c2q_tgt', 'c2a_src', 'c2a_tgt','entropy','entropy_a2q','entropy_c2q','entropy_c2a')
  data = transform(data, influencerScore=rowSums(data[,c('a2q_src','c2q_src','c2a_src')]))
  learnerScore = (data[,'a2q_tgt'])/replace(data[,'a2q_src'],data[,'a2q_src']==0,1)  # to avoide divergence
  data = transform(data, learnerScore=learnerScore)

  # Column-wise normalization of the data
  data.scaled = apply(data,2,scale)

  # Define filter and distance metric 
  data.dist = dist(data.scaled[,raw_vars])
  #model = prcomp(data.scaled[,raw_vars])  # PCA
  #data.filter = model$x[,1]   # First principal component
  data.filter = apply(as.matrix(data.dist),1,max)  # L-infinity eccentricity

  # TDA using Mapper
  data.mapper = mapper1D(distance_matrix = data.dist,
                         filter_values = data.filter,
                         num_intervals = 30,
                         percent_overlap = 70,
                         num_bins_when_clustering = 20)
  

  # Create resulting (plain) graph
  data.graph = graph_from_adjacency_matrix(data.mapper$adjacency, mode="undirected")
  mylayout=layout.auto(data.graph)
  plot(data.graph, layout = mylayout )  # plain graph
  
  
  # Mean value of the variable (of interest) for each vertex:
  names=c('a2q_src','a2q_tgt','c2q_src','c2q_tgt','c2a_src','c2a_tgt','deltaTS_mu','influencerScore','learnerScore','entropy','entropy_a2q','entropy_c2a','entropy_c2q')
  nVertices=data.mapper$num_vertices
  for (v in 1:length(names)){
    #mappingVar = data$a2q_src
    mappingVar = data[,names[v]]
    y.mean.vertex = rep(0,nVertices)
    nElementInVertex = rep(0,nVertices)
    for (i in 1:nVertices){
      points.in.vertex = data.mapper$points_in_vertex[[i]]
      nElementInVertex[i] = length(points.in.vertex)
      # variable which you want to mapping on TDA
      y.mean.vertex[i] = mean((mappingVar[points.in.vertex]))  
    }
    
    
    # Vertex size
    vertex.size = rep(0,nVertices)
    sizeVals = 5*seq(0,max(nElementInVertex),max(nElementInVertex)/20)/max(nElementInVertex)
    for (i in 1:nVertices){
      nElem = nElementInVertex[i] 
      sizeIdx = which.min(abs(sizeVals-nElem))
      vertex.size[i] = sizeVals[sizeIdx] + 6
      #vertex.size[i] = sqrt(sqrt(length((data.mapper$points_in_vertex[[i]]))))
    }
    
    
    
    # Mapper with colored node and vertex size proportional to the number of points inside:
    jet.colors = colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan",
                         "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
    y.mean.vertex.color = jet.colors(nVertices)[as.numeric(cut(y.mean.vertex,breaks = nVertices))]
    y.mean.vertex.grey = grey(1-(y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex) ))
    V(data.graph)$color = y.mean.vertex.color
    V(data.graph)$size = vertex.size
  
    fn_png = sprintf('%s/figures/TDA/rplot_y%d_%s_min%.0f_max_%.0f.png',proj_path,year,names[v],min(y.mean.vertex),max(y.mean.vertex))
    png(fn_png,width = 1024, height = 1024)
    plot(data.graph, main = names[v], layout = mylayout )
    legend(x = 'bottomright',
           legend = c("high","","","","","", "low"),pch=21,
           col="#777777", pt.bg=jet.colors(7)[7:1], pt.cex=2, cex=.8, bty="n", ncol=1)
    dev.off()
  }
}
