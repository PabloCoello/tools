function(data, formula){
    library('RJSONIO')
    res <- lm(formula=formula,
              data = data)
    return(toJSON(res))
}