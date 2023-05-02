Download Link: https://assignmentchef.com/product/solved-coms4036_7050a-lab-4-edges-corners-and-descriptors
<br>
In this lab, we will investigate edge detectors, corner detectors and image descriptors. In particular the Canny edge detector, Harris corner detector and histogram of oriented gradients descriptor.

<ul>

 <li>From Lab 1 you should have your 3 images of puzzle pieces with their own corresponding handmade masks. For this lab, you will need to read in one of these images and the corresponding mask to perform image processing operations on it.</li>

 <li>For question 1 and 2 you should downscale your image and mask for memory, computational and comparative reasons, from a size in pixels of 5120 × 3840 to 640 × 480, for question 3 you should downscale your images further to a size in pixels of 256 × 192.</li>

</ul>

<strong>Hint</strong>: you should use cv2.resize(img, (width, height)).

<strong>Note</strong>: Do not threshold your mask once downscaled for this lab, we want anti-aliased edges.

<ul>

 <li>Be wary of the datatype of your images being processed, silent errors can occur if you don’t cast an image to float32 from uint8 before processing, or threshold values can be incorrectly scaled. cv2.filter2D is one notable example of where things can go wrong.</li>

</ul>

<strong>Hint</strong>: While integer images have values in the range [0<em>,</em>255], float images should have values in the range [0<em>,</em>1]. A direct cast between datatypes will not preserve this relationship, instead have a look at skimage.img_as_float for when the image is read in.

<h1>1          Canny Edge Detector</h1>

In this section, you will implement your own version of the Canny edge detector, as well as investigate the effects of the algorithm’s different tuneable parameters.

1.1 Implement the Canny edge detection algorithm from scratch, your function should accept a greyscale image and three tuneable named parameters (defaults are given below): <em>σ</em>, <em>low threshold </em>and <em>high threshold</em>, similar in behaviour to that of the skimage canny edge detector.

<ul>

 <li>Remove any unnecessary noise by applying a Gaussian filter with <em>σ </em>= 4 to your greyscale image. Estimate the size of your Gaussian filter with the equation <em>size </em>= 2 · <em>radius </em>+ 1, where <em>radius </em>= <em>floor</em>(<em>truncate </em> <em>σ </em>+ 0<em>.</em>5), and <em>truncate </em>= 4<em>.</em>0 is the number of standard deviations away to truncate the filter.</li>

</ul>

<strong>Hint</strong>: Use nearest/replicate for the border padding – This is the same way skimage does it and we are going to compare later!

<ul>

 <li>Calculate the intensity gradient of the image, comprised of orientation and magnitudes, using vertical and horizontal Sobel filters.</li>

</ul>

<strong>Hint</strong>: use np.arctan2 instead of np.arctan to avoid incorrect values or divisions by zero, and np.hypot to avoid any potential precision errors. Use reflect for the border padding mode.

<ul>

 <li>Apply non-maximum suppression to get rid of any unwanted pixels which may not form part of an edge, leaving you with the “thin edges”.</li>

 <li>Apply double thresholding to determine potential edges. Use the weak/low threshold value of 0<em>.</em>1 and strong/high threshold value of 0<em>.</em>2.</li>

 <li>Track edges by Hysteresis, where weak edges are suppressed if not connected to strong edges.

  <ul>

   <li>Plot and label the results after each step in the Canny algorithm, include the blurred greyscale image (labelled with the estimated size of your Gaussian filter), horizontal and vertical Sobel filtered images, orientation and magnitude images, non-maximum suppressed image, double thresholded images, and the final result after Hysteresis. Make sure to label each image.</li>

  </ul></li>

</ul>

Figure 1: Example edges detected using Canny.

<ul>

 <li>Compare your final result to the skimage canny function (skimage.feature.canny) run with default arguments.</li>

</ul>

<strong>Hint</strong>: Read the docs<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> and tune your algorithm’s arguments so the outputs are as close as possible.

<ul>

 <li>Give reasons for the effect on the output after increasing or decreasing each parameter, including <em>low threshold</em>, <em>high threshold </em>and <em>σ</em>. Plot your different parameter traversals used to come to your conclusions. You may use the skimage.feature.canny function for this question.</li>

 <li>Adjust the parameters of your implementation to try and obtain the best or cleanest result. Plot this result with your chosen parameters. Again you may use the skimage.feature.canny function.</li>

</ul>

<h1>2          Harris Corner Detector</h1>

2.1 Your task is to implement the Harris Corner Detector, your function signature should accept a greyscale image with three additional tuneable named parameters (defaults are given below): <em>σ</em>, <em>κ </em>and <em>τ</em>, where <em>σ </em>is the standard deviation of the Gaussian filter representing the weight matrix, <em>κ </em>is the sensitivity factor that separates corners from edges and <em>τ </em>is the normalised response threshold.

<ul>

 <li>Compute your vertical and horizontal image derivatives from your greyscale image using the appropriately oriented Sobel filters and zero padding.</li>

 <li>Use these derivatives to compute your image structure tensors (<strong>S</strong><em><sub>ij </sub></em>in your textbook), weighted according to the Gaussian with default <em>σ </em>= 1. Again use zero padding.</li>

</ul>

<strong>Hint</strong>: Estimate the size of your Gaussian filter the same way you did for question 1.1. You do not need to compute and store the structure tensors for each pixel as matrices as shown in the textbook, it will be easier to decompose the matrix into its three unique components that can be convolved separately, while still being used as arguments for the next step.

<ul>

 <li>Calculate the responses (<em>c<sub>ij</sub></em>) from the structure tensors, with the default value for <em>κ </em>= 0<em>.</em>05.</li>

</ul>

<strong>Note</strong>: Do not use the erroneous formula from the textbook! Instead use:

<em>c<sub>ij </sub></em>= <em>λ</em><sub>1</sub><em>λ</em><sub>2 </sub>− <em>κ</em>(<em>λ</em><sub>1 </sub>+ <em>λ</em><sub>2</sub>)<sup>2 </sup>= det[<strong>S</strong><em><sub>ij</sub></em>] − <em>κ </em>· trace[<strong>S</strong><em><sub>ij</sub></em>]<sup>2</sup>

<strong>Hint</strong>: Following on from the previous hint, if you have the 2 × 2 matrix  the determinant is <em>ac </em>− <em>b</em><sup>2 </sup>and the trace is <em>a </em>+ <em>c</em>.

<ul>

 <li>Remove spurious corners from the responses by only keeping local maxima, do this by analysing the 8 neighbouring values of the centre value in a 3×3 sliding window and keeping the centre only if it is greater than its neighbours.</li>

</ul>

<strong>Hint</strong>: Be careful not to change the image itself while you are using the sliding window.

<ul>

 <li>Use the default threshold ratio <em>τ </em>= 0<em>.</em>05 to return the points where .</li>

</ul>

Figure 2: Example mask with corners detected.

<ul>

 <li>Apply your Harris corner detector to the mask image with default arguments and plot the detected corners by drawing circles around them.</li>

</ul>

<strong>Hint</strong>: draw a circle at <em>x,y </em>with cv2.circle(cornerim, (x, y), radius=12, color=(0,255,0), thickness=2)

<ul>

 <li>Test different values of <em>κ </em>∈ {0<em>.</em>025<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>2}, <em>σ </em>∈ {1<em>,</em>2<em>,</em>4<em>,</em>8} and <em>τ </em>∈ {0<em>.</em>01<em>,</em>0<em>.</em>05<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>2} with your implementation of the Harris corner detector. You do not need to run through all permutations of parameters, rather plot a traversal for each parameter and keep the remainder as the defaults (one plot for each parameter, with a labelled image corresponding to each parameter value in the set). Perform these tests on the greyscale version of the original image, not your mask. Draw the points on your images as in the previous question.</li>

 <li>How could you further reduce the number of points that are closely clustered together?</li>

</ul>

<h1>3          Histogram of Oriented Gradients Descriptor</h1>

For this section, you will use the RGB image of your original puzzle piece, but instead downscaled to a size of 256 × 192 pixels to be used for calculating the Histogram of Oriented Gradients (HoG) Descriptor. (The descriptor was originally designed to detect humans and operated on images at a size of 64 × 128 pixels, but we want pretty plots!)

3.1 Your final task is to implement the Histogram of Oriented Gradients Descriptor (HoG). Create a function that accepts an <strong>RGB </strong>image and three named arguments <em>orientations </em>= 9, <em>pixels per cell </em>= 8 and <em>cells per block </em>= 2, the cell size is the number of pixels along one axis of a square cell, and the block size is the number of cells along one axis of a square block.

<ul>

 <li>Make sure that the input image is valid to avoid any errors later. <strong>assert</strong><a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> that the width and height of the image in pixels is a multiple of <em>pixels per cell</em>, and that the width and height are at least <em>cells per block </em> <em>pixels per cell </em>pixels in size.</li>

 <li>Compute the orientation and gradient magnitude image, this is similar to question 1, however use the non-square first derivative filters, [−1<em>,</em>0<em>,</em>1] and [−1<em>,</em>0<em>,</em>1]<em><sup>T </sup></em>, on each channel in the RGB image with reflect padding. The final magnitude of the gradient at a pixel is the maximum of the magnitude of gradients of the three channels, and the angle is the angle from the channel corresponding to the maximum gradient.</li>

</ul>

<strong>Hint</strong>: the numpy functions np.argmax and np.take_along_axis may come in handy.

<ul>

 <li>Generate a histogram for each cell in the image by performing orientation binning using a weighted voting procedure. The weighted voting procedure splits the magnitude of the current pixel in the cell between the nearest two bins based on the ratio between the bins of the orientation of that pixel. Orientations are unsigned and are binned between 0<sup>◦ </sup>and 180<sup>◦</sup>, the number of bins is controlled by the <em>orientations </em>parameter (for example if there are 9 bins they are centred on ). Remember to use wrap-around so that bin 0<sup>◦ </sup>equals bin 180<sup>◦ </sup>(For example if a pixel’s orientation is 175<sup>◦ </sup>and its magnitude is 100, it contributes 25 to the total of bin 160<sup>◦ </sup>and 75 to the total of bin 0<sup>◦</sup>).</li>

</ul>

<strong>Hint</strong>: you can use a 3D array to store these histograms, treating the first two axes as the cell y &amp; x coordinates and the last as the histogram bins. Bonus points if you can calculate the histogram of a single cell purely with numpy operations and no python loops!

<ul>

 <li>Generate descriptor blocks by L2 normalising the concatenated histograms of the cells within a <em>cells </em><em>per block </em>× <em>cells per block </em>sliding window over all the cell histograms. Per axis there should be <em>cells </em><em>in axis</em>−(<em>cells per block </em>−1) blocks generated as they are overlapping. Normalisation is performed over the entire concatenated vector of cell histograms in the block ie. <em>normalised block </em>= <sub>||</sub><em><u><sup>block</sup></u><sub>block</sub></em><sub>||</sub>.</li>

</ul>

<strong>Hint</strong>: you can use slicing to extract a block of cells from your cell array and use np.flatten to obtain your unnormalised block vector/descriptor.

<ul>

 <li>Finally generate your image descriptor as the flattened vector of all your normalised block descriptors. This vector is meant to be quite large!</li>

</ul>

Figure 3: From left to right: max gradient magnitudes, corresponding max orientations, cell histogram maxes, block descriptor maxes. (All images are contrast stretched.)

<ul>

 <li>Plot the following steps of your HoG function called with default values as in figure 3, including the maximal gradient magnitude and orientation images (these images should be the original image in size), the image formed by taking the max value of each cell’s histogram, and lastly the image formed by taking the max value of each block’s normalised descriptor vector. Make sure to label and give the width and height of each image.</li>

</ul>

<strong>Hint</strong>: remember to contrast stretch your plotted images and use cmap=”inferno” for matplotlib. The reason we are taking the max over the cell histograms and the max over the block descriptors is not necessarily because that is useful in practice, but because otherwise there would just be too many images to put in your report. For your own interest have a go at plotting each bin/descriptor element separately as an image.

<ul>

 <li>For the previous question, what is the shape/size of your final image descriptor?</li>

 <li>Outline a method for using the HoG descriptor to detect puzzle pieces in images. Think about what training samples you would need, the shape and size of the images, how you would break up larger images, and how you would detect puzzle pieces at different scales.</li>

 <li>Is the HoG descriptor a good method for detecting puzzle pieces in the previous question, give reasons for why or why not – If not, can you suggest an alternative?</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny">https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <strong>assert </strong>is a python keyword useful for debugging code.