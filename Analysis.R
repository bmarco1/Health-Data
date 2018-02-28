
 analysis <- group_by(health, age)
 analysis1 <-summarise(analysis, count=n(), rate1 = mean(charges))
 analysis2 <- filter(analysis1, rate1 >= 10000)
 

 ggplot(analysis2, aes(age, rate1))+geom_point(aes(size = count, color = factor(rate1)), alpha=1/2)+
   theme(legend.position = "none")+
   labs(x = "age", y = "mean charges")
 
 
 ggplot(data = health) +
   geom_point(mapping = aes( x =age, y = charges, color = smoker)) +
   facet_wrap(~ children, nrow = 2)

 
 ggplot(data = health) +
   geom_bar(mapping = aes(x = smoker, y= ))
 
 ggplot(data = health) +
   geom_smooth(mapping = aes(x=charges, y=age, linetype = smoker))
 
 
 ggplot(data = health) +
   geom_point(mapping = aes( x =age, y = charges, color=smoker)) +
   facet_wrap(~ children, nrow = 2)
 
 
 ggplot(data = health) +
   geom_smooth(mapping = aes(x=charges, y=bmi, linetype = smoker))
 
 
 ggplot(data = health) +
   (mapping = aes(x = bmi, y = charges, color = smoker)) + 
   geom_point()+
   geom_smooth()
 
 ggplot(data = health) +
   geom_bar(mapping = aes(x = children, y= ))
 
 ggplot(data = health) +
   geom_bar(mapping = aes(x = sex, y= , fill = smoker))

 
 ggplot(data = health) +
   geom_point(mapping = aes( x =age, y = charges, color = sex)) +
   facet_wrap(~ children, nrow = 2)

 
 ggplot(data = health) +
   geom_point(mapping = aes(x=charges, y=age, color = region))
 
 ggplot(data = health) +
   (mapping = aes( x =bmi, y = charges, color = smoker)) +
   geom_point()+
   geom_smooth()+
   facet_wrap(~ region, nrow = 2)

 
 
 
 