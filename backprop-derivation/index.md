# Everything you need to know about Supervised Learning

I've stumbled across a lot of different resources to learn machine learning, which usually range from kind of decent to atrocious. I made this post to 1) help you quickly learn and gain intuitions, and 2) to reinforce my own knowledge of this area.

Here I cover all the basics of Neural Networks (NN), and teach you how to implement and train any architecture from scratch using Supervised Learning. I cover backpropagation calculations, and compute the specific example of a dense NN.

I made this a self-contained note. All you need to know is single variable calculus, and what a dot product is. 

*When I use a new word of ML jargon for the first time, I indicate this with quotes single quotes ‘ ’*

# Overview of how NNs are used and how they learn

It’s very important to have an overview before going into the details:

### Definition of a NN

A Neural Network (NN) is just a function. It acts on an input of numbers (a vector <img src="https://i.upmath.me/svg/%5Cvec%20x" alt="\vec x" />), to produce an output of numbers (another vector <img src="https://i.upmath.me/svg/%5Cvec%20y" alt="\vec y" />),

<img src="https://i.upmath.me/svg/NN(%5Cvec%20x)%20%3D%20%5Cvec%20y" alt="NN(\vec x) = \vec y" />
(We can be lazy with the vector symbol and write <img src="https://i.upmath.me/svg/x" alt="x" /> instead of <img src="https://i.upmath.me/svg/%5Cvec%20x" alt="\vec x" />).

Let's take the simplest case, a "dense NN". I'll describe how it works. Describing why it works/comparing it to neurons in the human brain is a whole other discussion.

The NN takes in x. It multiplies x by some matrix matrix <img src="https://i.upmath.me/svg/W" alt="W" /> to get <img src="https://i.upmath.me/svg/W%20x" alt="W x" />. The NN then acts on <img src="https://i.upmath.me/svg/W%20x" alt="W x" /> by some function <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" /> to get <img src="https://i.upmath.me/svg/%5Csigma(W%20x)" alt="\sigma(W x)" />. You can see the pattern - the NN keeps acting on the previous input, alternating between using a matrix or a function. We stop at some point, say after <img src="https://i.upmath.me/svg/d" alt="d" /> matrix multiplications, and the end result is our y. 

The matrices <img src="https://i.upmath.me/svg/W" alt="W" /> are different from each other, so we need to label them. Computer scientists start counting at 0, so the first weight matrix is indicated by <img src="https://i.upmath.me/svg/W%5E0" alt="W^0" />, second by <img src="https://i.upmath.me/svg/W%5E1" alt="W^1" />, and so on. The list of all weights is written as <img src="https://i.upmath.me/svg/W%3D%20(W%5E0%2CW%5E1%2C...%2CW%5E%7Bd-1%7D)" alt="W= (W^0,W^1,...,W^{d-1})" />. 

The <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" /> are usually all the same, so we don't have to label them.

This is fully described by:

![zNN diagram](https://user-images.githubusercontent.com/51329589/186346773-709e42a7-7996-4c1f-a5a9-5bf4bf8e29cf.jpg)

Lines indicate the action on each component of a vector. Ex. the line going from the <img src="https://i.upmath.me/svg/j" alt="j" />th element in the 1st layer to the <img src="https://i.upmath.me/svg/i" alt="i" />th element in the 2nd layer is <img src="https://i.upmath.me/svg/W%5E0_%7Bij%7D" alt="W^0_{ij}" />

Detail: If you're familiar with dense NNs, you might have noticed I didn't include any 'bias' terms here. They can actually be absorbed into the weights W, taking <img src="https://i.upmath.me/svg/x%5Crightarrow%20(1%2Cx)%20" alt="x\rightarrow (1,x) " />, <img src="https://i.upmath.me/svg/W%5Crightarrow%20(%5Cvec%20b%2CW)" alt="W\rightarrow (\vec b,W)" />. It's very easy to extend my analysis to have bias terms, and I'll have a section on this in the end.

A simpler way of writing this:

![zNN diagram 2](https://user-images.githubusercontent.com/51329589/186346634-80abf54e-ecae-485f-8e72-54843bf4f8b8.jpg)

Vectors are drawn as lines. Operators which act on vectors are drawn as boxes.

Clearly, the full description of this NN is

<img src="https://i.upmath.me/svg/NN(x%3B%20W%2C%5Csigma)%20%3D%20y" alt="NN(x; W,\sigma) = y" />

(We are usually lazy and don’t include the terms to the right of ‘;’, which are 'implicit').

Actually, this is a very general statement. We came up with the form of this function using the case of a dense NN, but _any_ well-performing NN must have an input (x) and both linear (W) and nonlinear <img src="https://i.upmath.me/svg/(%5Csigma)" alt="(\sigma)" /> parts. This includes Dense NNs (obviously!), Convolutional NNs, Transformers, Recurrent NNs, and so on. 

Linearity (W) is needed to teach NNs (we will see this during in the backpropagation section!). Nonlinearity <img src="https://i.upmath.me/svg/(%5Csigma)" alt="(\sigma)" /> are used between layers since they give way better results. You might have expected this, since linear functions are very simple and need something more.

### Teaching a NN

We know that NNs are just functions - a 'smart' NN is just really good at taking an input x to it's desired output y.

So what do we actually need to change about a NN to make it smarter?

Well, all the <img src="https://i.upmath.me/svg/%20%5Csigma%20" alt=" \sigma " />'s are usually the same, and the input x is whatever we want it to be. The only thing we can really change are the linear parameters W.

**So the only thing a neural network does when it learns is change its weights W**

We will ‘train’ our NN with a ton of data. This is called ‘Supervised Learning’:

1) Initialize the NN so its weights follow some distribution (bell curve/'Gaussian' distribution usually works the best). 
 
2) Gather a lot of inputs and their desired outputs <img src="https://i.upmath.me/svg/(x_i%2Cy_i)" alt="(x_i,y_i)" />. 

3) Input <img src="https://i.upmath.me/svg/x_i" alt="x_i" /> into the NN and tell the NN to change its weights so it outputs <img src="https://i.upmath.me/svg/y_i" alt="y_i" />. 

If we keep doing this, the NN would change its weights W more and more until all <img src="https://i.upmath.me/svg/x_i" alt="x_i" /> roughly give their corresponding <img src="https://i.upmath.me/svg/y_i" alt="y_i" />. (details in all their glory the next sections!)

### Why NNs are useful

The trained NN is then useful - if the NN learns from enough data, and doesn’t memorize or overgeneralize (‘overfit’ or ‘underfit’ the data), it can take any input X that it hasn’t seen before and accurately give us an output Y. You can see why this is useful if the inputs are market data and the output is Google's stock price at the end of the trading day. Or if the inputs are details of someone’s medical history and the outputs are their probability of developing diseases. (all these inputs and outputs are encoded as vectors). NNs apply very broadly. 

# Backpropagation

We now have the problem: given a set of 'training data' <img src="https://i.upmath.me/svg/S%3D(x_i%2Cy_i)" alt="S=(x_i,y_i)" />, find the weights W of a NN so that NN<img src="https://i.upmath.me/svg/(x_i%3BW%2C%5Csigma)%20%3D%20y_i" alt="(x_i;W,\sigma) = y_i" />. 

No human knows of a _great_ way to find these weights. The best approach we know of is called ‘Backpropagation’. It works, but at huge computational cost, and is certainly not what the human brain is doing. By the end of this, you’ll know all the details of Backpropagation.

### A mathematical statement of the problem - The Loss function

We need to describe our problem using mathematics.

The main idea: for each training pair <img src="https://i.upmath.me/svg/(x_i%2Cy_i)" alt="(x_i,y_i)" /> we want to pass <img src="https://i.upmath.me/svg/x_i" alt="x_i" /> through the NN and make the result as close to <img src="https://i.upmath.me/svg/y_i" alt="y_i" /> as possible, by changing W. In other words, we want to change W so that

<img src="https://i.upmath.me/svg/NN(x_i%2CW)%20%3D%20y_i" alt="NN(x_i,W) = y_i" />

We need a systematic way of doing this. Guessing and checking the weights would take forever.

The trick - turn this into a minimization problem. We want to find W that minimizes the function <img src="https://i.upmath.me/svg/L(%20NN(x_i%2CW)%2C%20y_i%20)%20%3D%20%7CNN(x_i%2CW)%20%E2%80%93%20y_i%7C" alt="L( NN(x_i,W), y_i ) = |NN(x_i,W) – y_i|" />. (if it is minimized then <img src="https://i.upmath.me/svg/NN(x_i%2CW)%20-%20y_i%20%3D%200" alt="NN(x_i,W) - y_i = 0" />, and obviously <img src="https://i.upmath.me/svg/NN(x_i%2CW)%20%3D%20y_i)" alt="NN(x_i,W) = y_i)" />. 

Let’s make this idea more general, and do this for all of the data points we have instead of just considering a single <img src="https://i.upmath.me/svg/(x_i%2Cy_i)" alt="(x_i,y_i)" /> pair. Define a ‘Total Loss Function’ to minimize:

<img src="https://i.upmath.me/svg/Loss(W%3B%20S)%20%3D%20%5Csum_%7BS%3D%7B(x_i%2Cy_i)%7D%7D%20L(%20NN(x_i%2CW)%2C%20y_i%20)" alt="Loss(W; S) = \sum_{S={(x_i,y_i)}} L( NN(x_i,W), y_i )" />

<img src="https://i.upmath.me/svg/Loss(W%3BS)" alt="Loss(W;S)" /> is just a single number that depends on the weights W and training set S. We want it to take the smallest value when the weights W fit our data S the best - when <img src="https://i.upmath.me/svg/NN(x_i%2CW)%20%3D%20y_i" alt="NN(x_i,W) = y_i" /> for all <img src="https://i.upmath.me/svg/(x_i%2Cy_i)" alt="(x_i,y_i)" />. I could have chosen any function that does this. I only chose to sum over S since sums are easy to work with and has some meaning (a product of so many terms would quickly vanish).

Note that we are free to pick the form of <img src="https://i.upmath.me/svg/L(%20NN(x_i%2CW)%2C%20y_i%20)" alt="L( NN(x_i,W), y_i )" />,

It can be <img src="https://i.upmath.me/svg/%3D%7C%20NN(x_i%2CW)%20-%20y_i%7C" alt="=| NN(x_i,W) - y_i|" /> as before, 

or <img src="https://i.upmath.me/svg/%3D%5Cfrac12%20(NN(x_i%2CW)%20-%20y_i)%5E2" alt="=\frac12 (NN(x_i,W) - y_i)^2" /> (mean squared error)

or <img src="https://i.upmath.me/svg/%3Dp%20%5Cln%20p" alt="=p \ln p" /> (cross entropy loss function, so that we maximize entropy). 

So <img src="https://i.upmath.me/svg/L(y%E2%80%99%2Cy)" alt="L(y’,y)" /> can be anything, as long as it is differentiable, and has a minimum when <img align="center" src="https://i.upmath.me/svg/y%3Dy'" alt="y=y'" />.

Clearly, if we minimize this Loss function, we will successfully have found the weights that fit our data! 
### Minimizing the Loss with Gradient descent

We have transformed the problem of training the weights into a minimization problem.

Now how do we minimize <img src="https://i.upmath.me/svg/%5Ctext%7BLoss%7D(W%5E0%2CW%5E1%2C%E2%80%A6%2CW%5E%7Bd-1%7D)" alt="\text{Loss}(W^0,W^1,…,W^{d-1})" /> and thus find the weights?

It's easier to first minimize a function of a single variable <img src="https://i.upmath.me/svg/f(x)" alt="f(x)" />, and extend this to multiple variables later. 

Here's a minimization algorithm:

1) Make an initial guess for the value of x that minimizes <img src="https://i.upmath.me/svg/f(x)" alt="f(x)" />. Call it <img src="https://i.upmath.me/svg/x_0" alt="x_0" />. 

2) Walk away from <img src="https://i.upmath.me/svg/x_0" alt="x_0" /> by some small distance <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" />, where the direction of <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" /> is chosen so that <img src="https://i.upmath.me/svg/f(x)" alt="f(x)" /> decreases.

3) Repeat (2.) until you're at a (local) minimum.

The problem is finding the direction of <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" /> in step 2. 

We also need to tune the step size so it's not too big (stepping over minimum) or too small (not stepping at all), but this is 'hyperparameter optimization', which I won't discuss here.

This problem might remind you of the Taylor Series/basic calculus,

![ztaylor](https://user-images.githubusercontent.com/51329589/186343654-39b08400-ab6d-4138-9250-59cafc27dea5.jpg)

We can check this series is valid by making sure both sides are equal at <img src="https://i.upmath.me/svg/%5Cdelta%20%3D%200" alt="\delta = 0" />

![ztaylor check](https://user-images.githubusercontent.com/51329589/186343749-e203c8b0-dd81-46c0-aba3-d5794a52040f.jpg)

If our step size  <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" /> is small, we see that the change in f going from <img src="https://i.upmath.me/svg/f(x_0)" alt="f(x_0)" /> to <img src="https://i.upmath.me/svg/f(x_0%2B%5Cdelta)" alt="f(x_0+\delta)" /> is
<img src="https://i.upmath.me/svg/df%20%3D%20f(x_0%2B%5Cdelta)-f(x_0)%20%3D%20%5Cfrac%7Bdf%7D%7Bdx%7D%7C_%7Bx_0%7D%20%5Cdelta%20" alt="df = f(x_0+\delta)-f(x_0) = \frac{df}{dx}|_{x_0} \delta " />

We want f to be decreasing, so <img src="https://i.upmath.me/svg/df" alt="df" /> should be negative. We can solve for  <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" />, <img src="https://i.upmath.me/svg/%5Cdelta%20%3D%20df%20%2F%5Cfrac%7Bdf%7D%7Bdx%7D%7C_%7Bx_0%7D" alt="\delta = df /\frac{df}{dx}|_{x_0}" />. So <img src="https://i.upmath.me/svg/%5Cdelta" alt="\delta" /> brings us towards a minimum if

<img src="https://i.upmath.me/svg/sign(%5Cdelta)%20%3D%20-%20sign(%5Cfrac%7Bdf%7D%7Bdx%7D%7C_%7Bx_0%7D)" alt="sign(\delta) = - sign(\frac{df}{dx}|_{x_0})" />

This is a totally obvious statement. It means if the slope of the line we are on is / shaped, we should walk to the left. If it is \ shaped, we should walk to the right.

-------------------------------------------------------------------------------------

Let's extend this to multivariable functions!

The previous algorithm and setup are exactly the same, except the initial guess and small step are vectors. I call them <img src="https://i.upmath.me/svg/%5Cvec%20r_0" alt="\vec r_0" /> and <img src="https://i.upmath.me/svg/d%20%7B%5Cvec%20r%7D" alt="d {\vec r}" />. I use a 3D space, but you can see that this extends to number of dimensions <img src="https://i.upmath.me/svg/(x%2Cy%2Cz%2Ct%2C...)" alt="(x,y,z,t,...)" />. 

![zhigh d taylor](https://user-images.githubusercontent.com/51329589/186745943-405e3412-67f8-40e3-88e5-1513618f2901.jpg)

In the blue box, I made an intuitive guess for the multivariable taylor series. If you've never seen these symbols before, <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20f(%5Cvec%20r)%7D%7B%5Cpartial%20x%7D" alt="\frac{\partial f(\vec r)}{\partial x}" /> is known as a 'partial derivative', and means "take the derivative of <img src="https://i.upmath.me/svg/f(x%2Cy%2Cz%2Ct%2C...)" alt="f(x,y,z,t,...)" /> with respect to x, holding all variables that aren't x (like y,z,t,...) constant".

We can confirm it is correct, similar to before (we can actually solve for all the terms in the series this way),

![zhigh d taylor check](https://user-images.githubusercontent.com/51329589/186746529-b802cb67-7d9b-4728-8efe-e80c9c4443d6.jpg)

Now we can find the change in f when moving from <img src="https://i.upmath.me/svg/%5Cvec%20r_0" alt="\vec r_0" /> to  <img src="https://i.upmath.me/svg/%5Cvec%20r_0%20%2B%20d%20%5Cvec%20r" alt="\vec r_0 + d \vec r" />. 

![zdf](https://user-images.githubusercontent.com/51329589/186344485-60769bc0-02b7-4a30-9dc3-9af31866b293.jpg)

If you've taken calculus before, you'll notice I defined the 'gradient' of a function of <img src="https://i.upmath.me/svg/%5Cvec%20r%20%3D%20(x%2Cy%2Cz%2Ct%2C...)" alt="\vec r = (x,y,z,t,...)" />. The gradient of <img src="https://i.upmath.me/svg/f(%5Cvec%20r)%20%3D%20f(x%2Cy%2Cz%2C...)" alt="f(\vec r) = f(x,y,z,...)" /> is just <img src="https://i.upmath.me/svg/%5Cnabla_%7B%5Cvec%20r%7D%20f(%5Cvec%20r)%20%5Cequiv%20(%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%2C%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D%2C%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%2C...)" alt="\nabla_{\vec r} f(\vec r) \equiv (\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z},...)" />. 

**Importantly, the gradient is in the direction of maximal change of <img src="https://i.upmath.me/svg/f" alt="f" />.** You can easily see this from the dot product - say you fix the magnitude of <img src="https://i.upmath.me/svg/d%20%5Cvec%20r" alt="d \vec r" />. Then <img src="https://i.upmath.me/svg/df" alt="df" /> will be largest when <img src="https://i.upmath.me/svg/d%20%5Cvec%20r" alt="d \vec r" /> is in the same direction as the gradient. So the gradient indeed points in the direction of maximum increase of the function. (Not surprisingly, you can see that <img src="https://i.upmath.me/svg/df" alt="df" /> will be 0 if <img src="https://i.upmath.me/svg/d%20%5Cvec%20r" alt="d \vec r" /> is perpendicular to the gradient)

We want to choose a direction <img src="https://i.upmath.me/svg/d%20%5Cvec%20r" alt="d \vec r" /> so that <img src="https://i.upmath.me/svg/df" alt="df" /> is the most negative. But this direction is just <img src="https://i.upmath.me/svg/-%20%5Cnabla%20f%7C_%7B%5Cvec%20r_0%7D" alt="- \nabla f|_{\vec r_0}" /> direction.

Clearly if we iterate
<img src="https://i.upmath.me/svg/%5Cvec%20r%20%5Crightarrow%20%5Cvec%20r%20-%20%5Ceta%20%5Cnabla%20f" alt="\vec r \rightarrow \vec r - \eta \nabla f" />
we'll arrive at a local minimum. Here <img src="https://i.upmath.me/svg/%5Ceta" alt="\eta" /> is some small number called the 'step size' or 'learning rate'.


With our Loss function, we can iterate
<img src="https://i.upmath.me/svg/%5Cvec%20W%20%5Crightarrow%20%5Cvec%20W%20-%20%5Ceta%20%5Cnabla%20Loss(W%3BS)" alt="\vec W \rightarrow \vec W - \eta \nabla Loss(W;S)" />
to find W that minimizes Loss(W)!


Note <img src="https://i.upmath.me/svg/%5Cvec%20W%3D(W%5E0%2CW%5E1%2C...%2CW%5E%7Bd-1%7D)" alt="\vec W=(W^0,W^1,...,W^{d-1})" /> and <img src="https://i.upmath.me/svg/%5Cnabla%20Loss(W%3BS)%20%3D%20(%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W)%7D%7B%5Cpartial%20W%5E0%7D%2C%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W)%7D%7B%5Cpartial%20W%5E1%7D%2C...%2C%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W)%7D%7B%5Cpartial%20W%5E%7Bd-1%7D%7D)" alt="\nabla Loss(W;S) = (\frac{\partial Loss(\vec W)}{\partial W^0},\frac{\partial Loss(\vec W)}{\partial W^1},...,\frac{\partial Loss(\vec W)}{\partial W^{d-1}})" />

You might be very concerned we'll get stuck on local minima and not find the global minimum. This is a major issue for gradient descent methods: If there's a flat landscape where the gradient vanishes ('barren plateau'), or a deep well, we will get stuck.

To help with this issue, there are more effective 'optimization' methods than just subtracting the gradient from W, most famously 'Adam' or 'Adagrad'. They use 'hyperparameters' or dummy parameters that are simply details of the calculation, and are not fundamental to the NN. For instance, we can vary the learning rate <img src="https://i.upmath.me/svg/%5Ceta" alt="\eta" /> over time. 

**But all modern optimization methods are gradient-based and require** <img src="https://i.upmath.me/svg/%5Cnabla%20Loss" alt="\nabla Loss" />. We will calculate this now.

### Approaching <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W%3BS)%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial Loss(\vec W;S)}{\partial W^n}" />; Stochastic gradient descent and the chain rule

To find <img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W)%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial Loss(\vec W)}{\partial W^n}" />, we could vary each matrix element in <img src="https://i.upmath.me/svg/W%5En" alt="W^n" /> and see how <img src="https://i.upmath.me/svg/Loss" alt="Loss" /> changes. But each of these calculations requires a forward pass. If <img src="https://i.upmath.me/svg/W%5En" alt="W^n" /> is a NxN matrix, we would need <img src="https://i.upmath.me/svg/N%5E2" alt="N^2" /> forward passes. This requires way too much computational power, and does not scale.

Instead, we need to find <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20Loss(%5Cvec%20W%3BS)%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial Loss(\vec W;S)}{\partial W^n}" /> a smarter way.

Note that Loss(W;S) depends on S, so it uses every data point in our training data. 

Instead of dealing with Loss, we want to deal with <img src="https://i.upmath.me/svg/L(y'%2Cy)" alt="L(y',y)" />, which only uses a single data point. 

**Here's a huge simplifying assumption, 'Stochastic Gradient descent': **

We can randomly/'stochastically' pick a data point <img src="https://i.upmath.me/svg/(x%2Cy)" alt="(x,y)" /> one at a time. Then we calculate the gradient <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L(NN(x%3B%5Cvec%20W)%2Cy)%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial L(NN(x;\vec W),y)}{\partial W^n}" /> ("L" instead of "Loss"). 

This gradient helps us decrease <img src="https://i.upmath.me/svg/L(NN(x%2CW)%2Cy)" alt="L(NN(x,W),y)" />, and thus <img src="https://i.upmath.me/svg/Loss(W%2CS)" alt="Loss(W,S)" />. (And we now see since we're only considering the "L" functions one at a time, and not the total Loss function, the form of the total Loss function (sum,product,etc.) won't affet training at all).

It's easiest to consider a general NN,

![zNN diagram bp](https://user-images.githubusercontent.com/51329589/186344800-03b0eb72-1a1c-4b18-9e79-46dfb52db435.png)

The boxes are either <img src="https://i.upmath.me/svg/W" alt="W" /> or <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" />. The last box is the loss <img src="https://i.upmath.me/svg/L" alt="L" />, so the NN outputs <img src="https://i.upmath.me/svg/L(NN(x%3BW)%2Cy)" alt="L(NN(x;W),y)" />.

Let's find the <img src="https://i.upmath.me/svg/L" alt="L" />th weight's partial (<img src="https://i.upmath.me/svg/L" alt="L" /> here being an index, not the loss function), 
<img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L(z%5Ed%2Cy)%7D%7B%5Cpartial%20W%5E%7B(L)%7D%7D" alt="\frac{\partial L(z^d,y)}{\partial W^{(L)}}" />. Note the loss <img src="https://i.upmath.me/svg/L(z%5Ed%2Cy)" alt="L(z^d,y)" /> is often written as <img src="https://i.upmath.me/svg/L(z%5Ed)" alt="L(z^d)" /> or even just <img src="https://i.upmath.me/svg/L" alt="L" /> with <img src="https://i.upmath.me/svg/y" alt="y" /> and <img src="https://i.upmath.me/svg/z%5Ed" alt="z^d" /> implicit. 

If the diagram was a dense NN, the dependencies would clearly be

<img src="https://i.upmath.me/svg/L(z%5Ed%2Cy)" alt="L(z^d,y)" />, or "L depends on <img src="https://i.upmath.me/svg/z%5Ed" alt="z^d" /> and <img src="https://i.upmath.me/svg/y" alt="y" />"

<img src="https://i.upmath.me/svg/z%5Ed(W%5E%7Bd-1%7D%2Cz%5E%7Bd-1%7D)" alt="z^d(W^{d-1},z^{d-1})" />

<img src="https://i.upmath.me/svg/z%5E%7Bd-1%7D(%5Csigma%2C%20z%5E%7Bd-2%7D)" alt="z^{d-1}(\sigma, z^{d-2})" />

<img src="https://i.upmath.me/svg/z%5E%7Bd-2%7D(W%5E%7Bd-2%7D%2Cz%5E%7Bd-3%7D)" alt="z^{d-2}(W^{d-2},z^{d-3})" />

If we wanted to take <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7Bd-2%7D%7D" alt="\frac{\partial L}{\partial W^{d-2}}" />, we would have a mess,

<img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20W%5E%7Bd-2%7D%7D%20L(z%5Ed(W%5E%7Bd-1%7D%2Cz%5E%7Bd-1%7D(%5Csigma%2C%20z%5E%7Bd-2%7D(W%5E%7Bd-2%7D%2Cz%5E%7Bd-3%7D))))" alt="\frac{\partial}{\partial W^{d-2}} L(z^d(W^{d-1},z^{d-1}(\sigma, z^{d-2}(W^{d-2},z^{d-3}))))" />

**The trick is to use the 'chain rule':**

We've seen,

<img src="https://i.upmath.me/svg/df(x%2Cy%2Cz%2C...)%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%7C_%5Ctext%7By%2Cz%7D%20dx%20%2B%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D%7C_%5Ctext%7Bx%2Cz%7D%20dy%2B%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%7C_%5Ctext%7Bx%2Cy%7D%20dz%20%2B%20..." alt="df(x,y,z,...) = \frac{\partial f}{\partial x}|_\text{y,z} dx + \frac{\partial f}{\partial y}|_\text{x,z} dy+\frac{\partial f}{\partial z}|_\text{x,y} dz + ..." />

Here <img src="https://i.upmath.me/svg/%7C_%5Ctext%7By%2Cz%7D" alt="|_\text{y,z}" /> means "hold y and z constant", and is usually not written but implied. 

But what if <img src="https://i.upmath.me/svg/x%2Cy%2Cz%2C..." alt="x,y,z,..." /> could depend on other variables? For example <img src="https://i.upmath.me/svg/x(t%2Cu)" alt="x(t,u)" />, <img src="https://i.upmath.me/svg/y(u)" alt="y(u)" />, <img src="https://i.upmath.me/svg/z(t%2Cu)" alt="z(t,u)" />.

Then similarly we have

<img src="https://i.upmath.me/svg/dx%20%3D%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20t%7D%20dt%20%2B%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20u%7D%20du" alt="dx = \frac{\partial x}{\partial t} dt + \frac{\partial x}{\partial u} du" />

<img src="https://i.upmath.me/svg/dy%20%3D%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20u%7D%20du" alt="dy = \frac{\partial y}{\partial u} du" />

<img src="https://i.upmath.me/svg/dz%20%3D%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D%20dt%20%2B%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20u%7D%20du" alt="dz = \frac{\partial z}{\partial t} dt + \frac{\partial z}{\partial u} du" />

Now if we wanted <img src="https://i.upmath.me/svg/df" alt="df" /> in terms of <img src="https://i.upmath.me/svg/t" alt="t" /> and <img src="https://i.upmath.me/svg/u" alt="u" />, we could plug these 3 equations into our original <img src="https://i.upmath.me/svg/df" alt="df" />

<img src="https://i.upmath.me/svg/df%20%3D%20%0A%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%20(%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20t%7D%20dt%20%2B%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20u%7D%20du%20)%0A%2B%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D%20(%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20u%7D%20du)%0A%2B%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%20(%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D%20dt%20%2B%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20u%7D%20du)%0A" alt="df = 
 \frac{\partial f}{\partial x} (\frac{\partial x}{\partial t} dt + \frac{\partial x}{\partial u} du )
+\frac{\partial f}{\partial y} (\frac{\partial y}{\partial u} du)
+\frac{\partial f}{\partial z} (\frac{\partial z}{\partial t} dt + \frac{\partial z}{\partial u} du)
" />

We can now find <img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D" alt="\frac{\partial f}{\partial t}" />. We freeze all variables that don't depend on <img src="https://i.upmath.me/svg/t" alt="t" />, so <img src="https://i.upmath.me/svg/du%20%3D%200" alt="du = 0" />. Then we divide both sides by <img src="https://i.upmath.me/svg/dt" alt="dt" />. This gives us

<img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D%20%3D%20%0A%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20t%7D%0A%2B%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D%0A" alt="\frac{\partial f}{\partial t} = 
 \frac{\partial f}{\partial x} \frac{\partial x}{\partial t}
+\frac{\partial f}{\partial z} \frac{\partial z}{\partial t}
" />

Clearly, we can expand out any dependencies like this. So <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D" alt="\frac{\partial f}{\partial t}" /> is the sum of all possible derivative chains starting from <img src="https://i.upmath.me/svg/f" alt="f" /> and ending at <img src="https://i.upmath.me/svg/t" alt="t" /> (this might even seem intuitive or obvious to you!).

**This is the chain rule.**

<img src="https://i.upmath.me/svg/%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D%20%3D%5Csum%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%20%5Cfrac%7B%5Cpartial%20x_1%7D%7B%5Cpartial%20x_2%7D%20...%20%5Cfrac%7B%5Cpartial%20x_%7BN-1%7D%7D%7B%5Cpartial%20x_N%7D%20%5Cfrac%7B%5Cpartial%20x_N%7D%7B%5Cpartial%20t%7D%20" alt=" \frac{\partial f}{\partial t} =\sum \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial x_2} ... \frac{\partial x_{N-1}}{\partial x_N} \frac{\partial x_N}{\partial t} " />

<img src="https://i.upmath.me/svg/%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D" alt=" \frac{\partial f}{\partial t}" /> doesn't depend on <img src="https://i.upmath.me/svg/x1%2Cx2%2C..." alt="x1,x2,..." />, or 'dummy variables'. All the dummy variables will be on the top of one partial and bottom of another, and will 'cancel' themselves out.

### Chain rule with components

Chain rule is for numbers that depend on numbers. But we're dealing with vectors that depend on other vectors.

To reduce vectors to numbers, we need to consider their individual components. Let's write components with subscripts, and continue writing indexes with superscripts. For example: <img src="https://i.upmath.me/svg/W%5E7%20_%7B2%2C4%7D" alt="W^7 _{2,4}" /> is the value in the 7th weight matrix's 2nd row and 4rd column (this is if we start counting with a 1).

Now we can apply chain rule: If we have one vector that depends on another vector <img src="https://i.upmath.me/svg/%5Cvec%20A(%5Cvec%20B)" alt="\vec A(\vec B)" />, we need to consider that all the components in <img src="https://i.upmath.me/svg/%5Cvec%20A" alt="\vec A" /> can depend on all the components in <img src="https://i.upmath.me/svg/%5Cvec%20B" alt="\vec B" />. The chain rule then gives (summed indices in red),

![zNN chain rule indices](https://user-images.githubusercontent.com/51329589/186345049-8a1fcd6f-f269-4e30-8963-b8e9104cfa0f.png)

Note that chains involve only the variables <img src="https://i.upmath.me/svg/z%5Ed%2Cz%5E%7Bd-1%7D%2C..." alt="z^d,z^{d-1},..." /> and a single W, since different W's and <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" />'s don't depend on eachother and won't produce other chains.

There are 4 different pieces in this product (<img src="https://i.upmath.me/svg/%5Cfrac%7Bd%20z'%7D%7Bd%20z%7D" alt="\frac{d z'}{d z}" /> is 2 possible pieces since it can involve <img src="https://i.upmath.me/svg/W" alt="W" /> or <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" />),

![zmodule calculations new](https://user-images.githubusercontent.com/51329589/186350732-d4d6f37e-ed90-49a8-be98-60b9d3e5c8ab.jpg)

Note I didn't write any sums, but it should be understood that we are summing over repeated indices that aren't i or j (dummy indexes), like the dot "." dummy index. Also, the <img align="center" src="https://i.upmath.me/svg/%5Cdelta_%7Bi%2Cj%7D" alt="\delta_{i,j}" />'s are "Kronecker Delta Functions", which are 1 if i=j, and 0 otherwise.

This approach is general, and works for whatever configuration of NN you want. You can even add different types of boxes. We easily can program a computer to put these blocks together and compute all the gradients.

However, for the sake of understanding what this involves, let's consider a specific case and combine these pieces by hand!

### Back-of-envelope derivation of <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial L}{\partial W^n}" /> for a dense NN

Here's a recap of what we've done so far, assuming a dense NN.

![zdense NN diagram](https://user-images.githubusercontent.com/51329589/186347655-0256495b-1a25-4a30-acae-172d39bab8fb.jpg)

I use two variable names <img src="https://i.upmath.me/svg/x%5En%2Cy%5En" alt="x^n,y^n" /> to make it clear we're alternating between W and <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" />.


Here's a back-of-the-envelope calculation without indices,

![zback of envelope chain rule](https://user-images.githubusercontent.com/51329589/186345444-5ddb77b5-292c-4376-bc28-7502b22ba1aa.jpg)


Two main points:

1) We've written the gradient as a matrix multiplication, which is easy to calculate. (We need to consider the components to get an exact result - this result isn't fully correct)

2) <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7Bn-1%7D%7D" alt="\frac{\partial L}{\partial W^{n-1}}" /> is the same as <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7Bn%7D%7D" alt="\frac{\partial L}{\partial W^{n}}" />, except it has 2 extra terms (these terms shown in blue).

(2) means we can compute the rightmost partial, and use that result when calculating the partial to the left of that, and so on.

**So it is better to calculate partials going backwards - Backpropagation! (from chain rule, this is true in general)**


### Formal derivation of <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5En%7D" alt="\frac{\partial L}{\partial W^n}" /> for a dense NN

We know what to expect for a dense NN. Let's calculate!

We could brute-force compute the derivatives. But let's take advantage of the repeating structure.

![dense NN recursive approach](https://user-images.githubusercontent.com/51329589/186345533-804811a4-f48f-4369-8deb-03950dfc6da3.jpg)

I wrote it this way so that we can recurse on the repeating part (again in blue).

We can calculate this with indices,

![zdense NN indices part11](https://user-images.githubusercontent.com/51329589/186349090-6ff0d1e6-feea-421b-b52f-e67a9bb5122e.jpg)

It's easier for humans to work with vectors/matrices instead of indices, but working with matrices vs. vectors does not affect computation.

We have <img src="https://i.upmath.me/svg/%5B%5Cvec%20a%20%5Ccdot%20%5Cvec%20b%5ET%20%5D_%7Bi%2Cj%7D%20%3D%20a_i%20b_j" alt="[\vec a \cdot \vec b^T ]_{i,j} = a_i b_j" /> (just take <img src="https://i.upmath.me/svg/%5Cvec%20a" alt="\vec a" /> <img src="https://i.upmath.me/svg/%5Cvec%20b%5ET" alt="\vec b^T" /> and multiply them together. You'll get a matrix with those components).

So we can write this as a matrix,

![zdense NN indices part12](https://user-images.githubusercontent.com/51329589/186348871-002b1ff1-a7a3-4258-80c3-5be66a3a2c7d.jpg)


Calculating the other parts similarly,

![zdense NN indices part2](https://user-images.githubusercontent.com/51329589/186345901-a73e1af8-b50e-4dd5-8639-f8fc43a9929f.jpg)


Simplifying further,

![zalgorithm prep](https://user-images.githubusercontent.com/51329589/186592738-850c852b-f044-4d10-8827-c97a4dc91c9b.jpg)

### Gradient Descent Methods for dense NN

We now have an algorithm for calculating gradients given a training point <img src="https://i.upmath.me/svg/(x%2Cy)" alt="(x,y)" />, which indeed runs backwards:

![zalg1](https://user-images.githubusercontent.com/51329589/186847819-944c969d-f121-4769-a84e-699082ed6c99.png)

If we're using naive stochastic gradient descent (SGD) as our optimizer we could do <img align="center" src="https://i.upmath.me/svg/%5Cvec%20W%20%5Cleftarrow%20%5Cvec%20W%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cvec%20W%7D" alt="\vec W \leftarrow \vec W - \eta \frac{\partial L}{\partial \vec W}" /> at the very end. More effective is subtracting each component when first possible, so the changes in the weights are considered in the next step,

![zalg2](https://user-images.githubusercontent.com/51329589/186847864-2e1aa00f-af6d-4119-8672-80fa409b7f34.png)

Note that 'Adam' and 'Adagrad' optimizers are preferred to naive SGD, but we can't use a trick like this since their steps require the entire gradient.

### Training a NN

We have a general approach to compute the gradient <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cvec%20W%7D" alt="\frac{\partial L}{\partial \vec W}" /> and change all the weights <img src="https://i.upmath.me/svg/%5Cvec%20W" alt="\vec W" />, given a single training point <img src="https://i.upmath.me/svg/(x%2Cy)" alt="(x,y)" />. 

We need to iterate this over our training set. Every time we use the entire training set is known as an 'epoch'. 

The modern method for Supervised learning is:

```
for _ in num_epochs:
     training_data = RandomSort([(x1,y2), (x2,y2),...]) //randomly sort+select part of the training set every epoch
     for x,y in training_data:
          // compute the gradient (as above)
          // change the weights (pick one: naive SGD, Adam, Adagrad,...)
          // optional: change the hyperparameters (learning rate,...)
```

Stochastically sorting training data and using many epochs helps avoid getting stuck on local minima. Still, **the hardest part about training NNs is fine tuning the hyperparameters like the learning rate, so we don't get stuck.**


### Testing a NN

We've trained our NN over many epochs, and have a list of the weights <img src="https://i.upmath.me/svg/%5Cvec%20W" alt="\vec W" /> at every epoch.

How do we know which epoch's <img src="https://i.upmath.me/svg/%5Cvec%20W" alt="\vec W" /> to use?

It's useful to boil the accuracy/error of our NN down to a number. Anything that does this is called a 'metric'. For instance, the Loss can be used as a metric.

Should we then pick the <img src="https://i.upmath.me/svg/%5Cvec%20W" alt="\vec W" /> with the lowest error on the training set?

The training error typically looks like,

![Training vs testing 1](https://user-images.githubusercontent.com/51329589/186795760-2444271b-86f1-415d-9f36-5d5867955e48.jpg)

It increases and decreases (think a ball rolling up a hill with a deep hole at the top), slowly finding lower and lower minima. (It can roll up since one data point might want W to change one way, and another point might want it to change the other way). With infinite epochs, the NN should find the global minimum and have 0 training error. But this would be memorization/overfitting. Clearly NN can learn to overfit/underfit/fool us if we test only on training data.

From the beginning, we should split our data into a 'training set', and an independent 'testing set' the NN only sees during testing. 

The training and testing errors look like,

![Training vs testing 2](https://user-images.githubusercontent.com/51329589/186795787-386d0cd2-61bd-4ab6-8a99-94c2a2ffd7b8.jpg)


**Our ideal NN is the one with the lowest testing error (blue point)**

So if we don't think the testing error will decrease anymore, we should stop training our NN.

### Biases

As I promised in the beginning, we need to include the bias terms into our weights. Let's look at which of our 4 'blocks' get affected. 

I'll indicate the old terms with a tilde "~", and will write the new terms normally.

![Bias term 1](https://user-images.githubusercontent.com/51329589/186797008-b0805415-d411-442b-a01c-0cf6dcc1b5ab.jpg)

The weight 'block' changes, but in an easy way. 
As long as we take <img src="https://i.upmath.me/svg/%5Cvec%20x%5En%5Crightarrow%20(%5Cvec%20x%5En%2C1)%20" alt="\vec x^n\rightarrow (\vec x^n,1) " />, <img src="https://i.upmath.me/svg/W%5En%5Crightarrow%20(W%5En%2C%5Cvec%20b)" alt="W^n\rightarrow (W^n,\vec b)" />, we'll have <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20x'%7D%7B%5Cpartial%20x%7D%20~%20W" alt="\frac{\partial x'}{\partial x} ~ W" />. This is the form we had before. So the calculations we did still hold, just for slightly different <img src="https://i.upmath.me/svg/W%5En" alt="W^n" /> and <img src="https://i.upmath.me/svg/%5Cvec%20x%5En" alt="\vec x^n" />.

None of the other blocks change,

![Bias term 2](https://user-images.githubusercontent.com/51329589/186802595-cd581146-c203-443d-b388-7f5baa3fb9be.jpg)

So to implement biases, we just need to take all the inputs to the weights matrices to be <img src="https://i.upmath.me/svg/(%5Cvec%5Ctilde%7Bx%7D%7D%5En%2C1)%20" alt="(\vec\tilde{x}}^n,1) " />, and the weights to be <img src="https://i.upmath.me/svg/(%5Ctilde%7BW%7D%5En%2C%5Cvec%20b)" alt="(\tilde{W}^n,\vec b)" />

Note that we cannot learn the activation function like we learn the weights (how do you take <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Csigma%7D" alt="\frac{\partial L}{\partial \sigma}" />?) But in a way we *are* learning the activation function - the bias term is just shifting the function left or right (<img src="https://i.upmath.me/svg/%5Csigma(%5Ctilde%7BW%7D%20%5Cvec%20x%20%2B%20%5Cvec%20b)" alt="\sigma(\tilde{W} \vec x + \vec b)" />).

# Conclusion and summary

**The process of training an arbitrary NN with Supervised Learning:**

1) Differentiate all the 'blocks' to get <img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20z'%7D%7B%5Cpartial%20z%7D" alt="\frac{\partial z'}{\partial z}" />

2) Plugging these differentiated blocks into chain rule, compute the form of the partials <img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B(n)%7D%7D" alt="\frac{\partial L}{\partial W^{(n)}}" /> (involving matrix and vector products). Computers can do this easily with brute-force methods. 

3) For a training point (x,y), compute the the gradient <img align="center" src="https://i.upmath.me/svg/%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cvec%20W%7D%20%3D%20%5B%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B(0)%7D%7D%20%2C%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B(1)%7D%7D%2C...%2C%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%5E%7B(d-1)%7D%7D%5D" alt="\frac{\partial L}{\partial \vec W} = [\frac{\partial L}{\partial W^{(0)}} ,\frac{\partial L}{\partial W^{(1)}},...,\frac{\partial L}{\partial W^{(d-1)}}]" />, using the result of (2).

4) Use the gradient to change the weights

5) Repeat 3-4 over many training points (x,y), and over many epochs

**The freedoms you have in doing this are:**

0) pick the inputs, outputs, data, and architecture of the NN

1) pick the activation function <img src="https://i.upmath.me/svg/%5Csigma" alt="\sigma" /> (RelU, Sigmoid,...)

2) pick the form of L(y,y') (root mean squared error, mean squared error, cross entropy,...)

2.5) optional: add a regularizer term

3) pick the optimizer that applies the gradient to the weights (SGD, Adam, Adagrad,...)

3.5) optional: finetune the hyperparameters yourself

4) pick a metric to measure testing error (We have the same selection of functions as for the loss L(y,y'). We can use the loss as a metric, or something different)

You now know the basics about computing and training NNs. If you wanted, you could now train any NN architecture from scratch, if you had data. You also know pretty much all of the basic terminology and concepts in machine learning.

# Future posts:
I will soon post about transformers: their intuitions, uses, and differentiable blocks as done here.
I will also calculate the CNN differentiable block.
