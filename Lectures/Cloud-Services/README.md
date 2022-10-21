# Cloud Services

Main goal is to give students a sense of why they might need cloud services and, when they do, what their options are.

## Prerequisites

The level-ups section for this lecture asks the instructor to demonstrate:

- uploading notebooks to databricks and to colab
- pickling a model
- connecting to an AWS S3 bucket

Ideally, instructor will have the relevant accounts set up for this purpose.

If you have a particular form of deployment that you want to demonstrate (cloud function, Flask app, Dash app, Streamlit app) feel free to do this during the demo time at the end!
 
## Learning Goals

Learning goals listed at the top:

 - Explain the general concept of "the cloud";
 - Understand the cases where ***hardware acceleration*** is useful;
 - Understand the cases where ***cloud storage*** and the **Boto3** library in particular are useful;
 - Explain the purpose of ***deploying*** a machine learning model, particularly with the **Flask** library.

Summary at the end of the main lecture content:

* Cloud services are useful for **computationally intensive or long-running tasks**
* The major providers of cloud services are **Amazon Web Services (AWS), Microsoft Azure, and Google Cloud**
* As a data scientist, you will generally use cloud services to **get more computing power and/or to deploy machine learning models**
* If you want to get more computing power, consider:
  * Cloud instances/containers with GPUs, particularly **EC2**
  * Cloud notebooks, particularly **Google Colab**
  * Cloud storage, particularly S3 bucket storage with **Boto3**
* If you want to deploy a machine learning model, first pickle the model, then consider:
  * Deploying a model as an API, using either a **cloud function** or a minimal **Flask** app
  * Deploying a model as a full-stack web app, either using Flask or **Dash**

## Lecture Materials

[Jupyter Notebook: Cloud Services](cloud_services.ipynb)

## Lesson Plan

Provide a breakdown of the lecture activities using the structure below. 

### Introduction (10 Mins)

What the cloud is (just someone else's computer!) and why you might want to use cloud services (three main reasons: hardware acceleration, large-file storage, model serving).

What the major cloud providers are, particularly AWS, Azure, and Google Cloud.

Higher-performing cloud services tend to be more expensive, although they can get free credits to try them out.

### Hardware Acceleration (10 Mins)

Make the contrast between using packages like NumPy (software acceleration) and uploading your whole notebook to Google Colab (hardware acceleration).

Explain that cloud instances like EC2 can be more powerful than cloud notebooks, but require more comfort with the terminal/systems administration tasks.

#### Level Up: Cloud Notebook Demo (Optional, 5+ Mins)

Demo importing notebooks into other platforms. (Should be a familiar concept from the big data lectures.)

If you have an example of a model training that usually takes a long time (e.g. grid search) and you have credits to spare, you can show that in SageMaker.

### Cloud Storage (10 Mins)

As usual, AWS is a leading player here. S3 buckets are very popular with FI students (and cheap). Distinction between S3 buckets and cloud databases. Boto3 is the Python library used to connect to an S3 bucket.

### Model Persistence/Pickling (5 Mins)

This may or may not be a review for students. (There was a Canvas lesson previously.) But even if it is, good refresher on using `joblib` to save and to load objects, especially including fitted models.

#### Level Up: Pickling a Model for Deployment Demo (Optional, 5 Mins)

Walking through the actual code to pickle a model with `joblib`.

### Deploying a Model as an API (10 Mins)

Think back to Phase 1 when we learned about APIs. A server is running somewhere, accepts queries, returns JSON. You can deploy an ML model like that.

Discuss cloud functions and Flask apps as approaches for this.

### Deploying a Model as a Full-Stack Web App (10 Mins)

Instead of just an API interface, it's fun to make something with a web interface and interactive components.

Discuss Flask apps and Dash apps as approaches for this.

#### Level Up: Demo an App (Optional, 5 Mins)

INSTRUCTOR-SPECIFIC FOR NOW

No content for this appears in the notebook, but feel free to demo your own cloud function or app here.

### Conclusion (5 Mins)

Let students know that it will be possible to get support for flask for future projects (esp. including capstone). "Thanks, everyone!"

### Level Up: AWS S3 Buckets with Boto3 Demo (10 Mins)

Demonstrating how to connect to S3 buckets from a local Jupyter notebook. Key packages here are `boto3` and `s3fs`.

This level-up assumes that students understand the `joblib` logic (see *Level Up: Pickling a Model for Deployment Demo*) so it probably only makes sense to include at the end, time permitting.

## On the Flatiron S3 Bucket for the Data Science Curriculum Team

FIS has an S3 bucket for our use to store assets for the curriculum.

The root bucket is:
**curriculum-content**
our dedicated folder is:
**curriculum-content/data-science/**
Within the data-science folder there are currently 3 folders:
/data
/images
/models

Navigation
Navigation and exploration within the S3 bucket can be accomplished multiple ways, but one that works very well is the AWS CLI tool.
The AWS CLI tool can be downloaded from [HERE](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

### To connect to the bucket using `boto3`:

Go to https://aws.amazon.com and set up an AWS account with your Flatiron email. You will need to attach a credit card, but it definitely shouldn't be charged if you are only using this to access public content. Then from that account you'll need to get an access key and secret key. Then you can use the environment variable approach to tell `boto3` those credentials: Depending on your shell language, that means adding these lines to either a Bash config file (e.g. .bash_profile) or a Z shell config file (e.g. .zshrc)
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
and adding your actual credentials inside the quotes.

Then you probably want to restart the terminal and re-launch Jupyter Notebook to make sure those variables are loaded.

Then the AWS lines in the notebook should run without errors!
