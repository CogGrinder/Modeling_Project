## Pseudo code for optimal value
### greedy brute force strategy
#### ``p_x``
Algorithm : greedy brute force strategy to find the optimal value of ``p_x``
Input     : ``n``, ``m``, ``l(p_x)`` function

```python
p_x     <- -ceil(n/2)
p_x_min <- p_x
l_min   <- MAX_FLOAT
l_list  <- list of size n
while p_x < floor(n/2) :
    l <- evaluate l(p_x)
    if l_min > l :
        l_min = l
        p_x_min = p_x
    l_list[p_x] <- l
    p_x <- p_x + 1
return p_x_min, l_list
```

#### ``(p_x, p_y)``
Algorithm : greedy brute force strategy to find the optimal value for ``(p_x, p_y)``
Input     : ``n``, ``m``, ``l(p_x,p_y)`` function
```python
p_x     <- -ceil(n/2)
p_y     <- -ceil(m/2)
p_min   <- [p_x,p_y]
l_min   <- MAX_FLOAT
l_list  <- double list of size n x m
while p_x < floor(n/2) :
    while p_y < floor(m/2) :
        l <- evaluate l(p_x,p_y)
        if l_min > l :
            l_min = l
            p_x_min = p_x
            p_y_min = p_y
        l_list[p_x][p_y] <- l
        p_y <- p_y + 1
    p_x <- p_x + 1
return p_x_min, l_list
```
### non differentiable coordinate descent strategy 
Algorithm : non differentiable coordinate descent strategy to find the optimal value for ``(p_x, p_y)``
Input     : ``n``, ``m``, ``p_0`` a list of two real numbers, ``alpha_0`` the initial percentage, ``l(p_x,p_y)`` loss function,
 ``epsilon`` the stopping level for ``alpha``
```python
alpha     <- apl_0
p_x     <- p_0[0]
p_y     <- p_0[1]
l   <- MAX_FLOAT
while alpha < epsilon:
    l_p_x_over  <- evaluate l(p_x*(1+alpha),p_y)
    l_p_x_under <- evaluate l(p_x*(1-alpha),p_y)
    l_p_y_over  <- evaluate l(p_x,p_y*(1+alpha))
    l_p_y_under <- evaluate l(p_x,p_y*(1-alpha))
    
    if l_p_x_over < l :
        l = l_p_x_over
        p_x = p_x*(1+alpha)
        alpha <- alpha*1.1
    else if l_p_x_under < l :
        l = l_p_x_under
        p_x = p_x*(1-alpha)
        alpha <- alpha*1.1
    else :
        alpha <- alpha*0.5
    S
    if l_p_y_over < l :
        l = l_p_y_over
        p_y = p_y*(1+alpha)
        alpha <- alpha*1.1
    else if l_p_y_under < l :
        l = l_p_y_under
        p_y = p_y*(1-alpha)  
        alpha <- alpha*1.1
    else :
        alpha <- alpha*0.5



    p_y <- p_y + 1
    p_x <- p_x + 1
return p_x_min
```