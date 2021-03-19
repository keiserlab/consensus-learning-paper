## BlobOrNot

Source code for blobornot web apps.

See AWS_deployment_documentation.pdf for step by step AWS Elastic Beanstalk set up.
 
```
application.py
    Main script, contains all backend logic, database manipulation.

config.py
    Configuration file, specify csv settings

default_config.py
    Configuration file that loads database and S3 connection environment variables

password.npy
    Store user login credentials.

data/blobornot_examples.csv
    Images and data for all examples to be annotated, the order is the order for annotation, so you should shuffle the list before deployment. 
    
    There are four columns: 
        blobs is the path to 20x images,
        10xfields is the path to the 10x field tiles, 
        coords is the coordinates (x,y,w,h) of the “blobs” image in “10xfields”
        rotation is how many degrees the images should be rotated when displayed ie. 0, 90, 180 or 270
    Note: Img paths should be the path in the AWS S3 bucket, without bucket name.

requirement.txt
    Contains the library names and versions that are used in the application
 
backup_database.py
	Used to backup annotation records into a .csv file. Can be set to run regularly for daily backup.

set_password.py
	Set login credentials and unique id for users.

delete_all_annotations.py
	Clear the database of all annotations.

zipforaws.sh
	Zip all necessary files for Elastic Beanstalk app

Subfolders:
database
	Contains files for database. The models.py defines the tables in the database.
static
	Contains static images that are used in the application.
templates
	Contains all HTML templates. 
data
	Contains csvs with blob data needed to display example for annotation

```