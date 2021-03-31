

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/result_calibration.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/image_filter_test1.jpg "Binary Example"
[image4]: ./output_images/prespective_test1.jpg "Warp Example"
[image5]: ./output_images/lane_finder_test1.jpg "Fit Visual"
[image6]: ./output_images/projected_lane_test1.jpg "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `./notebook/Calibration.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image the code located at ./notebook/Filter.ipynb, I took serval step to get the result:
* use HLS color space 
* use threshold for S & L channel *color2_\*_threshold*, *color1_\*_threshold*
* apply sobel magnitude `which compute the absolute value of sobel gradient on x-axis`, direction `which compute the direction angle between both sobel gradient in x-axis and y-axis `, absolute thresholds `which compute the absolute value of magnitude for both axis of sobel gradient `
*
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform in `./notebook/Perspective.ipynb` includes a line 
```python
bird_image = camera.get_bird_view(undist)
``` 
in the 5th cell.  The `camera.get_bird_view()` function takes as inputs an image (`car_view image`), as well as source (`src`) and destination (`dst`) points are already passed to `Camera` class in 4th cell.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Given the warped binary image from the previous step I did the following `laneFinder.find_pixels()`:
* Calculate a histogram of the bottom half of the image
* Partition image to `nWindow` which passed to constructor through dictionary
* Starting from the bottom of the image, make rectangle with width 200, around the 1st peak and second peak of histogram
* move up and find pixels that are part of the sliding window on the 1st and 2nd peak
* return 1st peak and 2nd peak points as left and right lane points

Then I used these points to fit polynomial `lane_finder.fit_poly()`

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Given the result of the previous step I did the following `lane_finder.process_video()`:
* Compute `ploty` which is all y-axis range
* Compute `leftx` and `rightx` using the fitted_polynomial for both `left_fit` and `right_fit` respectively
* Multiply each of `leftx` and `rightx` by scale which is 3.7 / 700 *this number are choosen according to usa street rules* and Multiply `ploty` by scale 30 / 720 to convert them to meter
* Fit new polynomial using the new points from previous step `lane_finder.fit_poly()`
* Calculate curvature using the fitted line in meter using the formula in `lane_finder.find_radius()`
To Compute the vehicle position with respect to center, I did the following `lane_finder.find_vehicle_position()`:
* Get maximum y value
* Compute both left_lane and right lane x-axis value using fitted polynomial line
* Compute the average between both values
* Compute the difference between average value from previous step and the half of the width of the image
* Convert it to meter by multiplying it by scale 3.7/300

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `./notebook/Project_Lane.ipynb` in line
```python
 projected_image = camera.get_projected_image(undist, bird_view, result['ploty'], result['leftx'], result['rightx'])
```
this function take the `undist` undistorted image to project the lane on it, `bird_view` warped image to create overlayed image as same as current warped image, `ploty` that contain all y point, `leftx` all left lane points, `rightx` all right lane points.
This Function recast the `leftx`, `rightx` and `ploty` to be used in opencv `cv2.fillPoly()` then convert the filled_image in bird_view is converted to car_view using the inverse prespective transformation from bird view to car view using `cv2.warpPerspective()` 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the pipeline have some issue 
* Alot of hyper-parmeter tuning which can be solved using neural network segmentation techniques
* Is not robust enough when dealing with very sharp turn 
* Can easily be fooled by human moving on lane 