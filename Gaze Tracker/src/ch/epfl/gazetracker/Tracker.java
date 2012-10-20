package ch.epfl.gazetracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

public class Tracker {
    private static final String   TAG = "Tracker";
    
    private Mat mGray;
    private Mat mGrayClone;
    
    private String direction;
    
    private boolean savingImage = false;
	
	private CascadeClassifier 	  faceClassifier;
    private CascadeClassifier     eyeClassifier;
    private CascadeClassifier     noseClassifier;
	
    private static final Scalar   FACE_RECT_COLOR = new Scalar(0);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(51);
    private static final Scalar   PUPIL_COLOR = new Scalar(255);
    
    private Paint paint;
    
    public Tracker(Context context) {
    	/////Let's load the classifier...
    	try {
    		///// Load the face classifier
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            File bufferFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
            FileOutputStream os = new FileOutputStream(bufferFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            faceClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (faceClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                faceClassifier = null;
            } else {
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            }
            
    		///// Load the eye classifier
            is = context.getResources().openRawResource(R.raw.haarcascade_eye);
            bufferFile = new File(cascadeDir, "haarcascade_eye.xml");
            os = new FileOutputStream(bufferFile);

            buffer = new byte[4096];
            bytesRead = 0;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            eyeClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (eyeClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyeClassifier = null;
            } else {
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            }
            
    		///// Load the nose classifier
            is = context.getResources().openRawResource(R.raw.haarcascade_mcs_nose);
            bufferFile = new File(cascadeDir, "haarcascade_mcs_nose.xml");
            os = new FileOutputStream(bufferFile);

            buffer = new byte[4096];
            bytesRead = 0;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            noseClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (noseClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                noseClassifier = null;
            } else {
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            }
            
            cascadeDir.delete();            
            
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    	
    	direction = "";
    	
    	mGray = new Mat();
    	mGrayClone = new Mat();
    	
    	paint = new Paint();
        paint.setColor(Color.YELLOW);
        paint.setTextSize(50);
    }
    
    protected Bitmap processFrame(VideoCapture capture) {
		long tss = System.currentTimeMillis();
		
        long ts = System.currentTimeMillis();
        //capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);      
        mGrayClone = mGray.clone();
        long te = System.currentTimeMillis();
        Log.d(TAG, "Retrieving pictures from camera : " + (te - ts) + " ms.");
        
        direction = "";
        
        ts = System.currentTimeMillis();
        Rect[] facesArray = detectFaces(mGray);        
        te = System.currentTimeMillis();            
        Log.d(TAG, "Face detection : " + (te - ts) + " ms.");
        Log.d(TAG, facesArray.length + " faces detected.");
        
        for (int i = 0; i < facesArray.length; i++) {
            Mat faceMat = mGray.submat(facesArray[i]);
            
            ts = System.currentTimeMillis();
            Rect nose = detectNose(faceMat);
            te = System.currentTimeMillis();
            Log.d(TAG, "Nose detection : " + (te - ts) + " ms.");
            
            if (nose != null) {
            	ts = System.currentTimeMillis();
                Rect[] eyesArray = detectEyes(faceMat);
                te = System.currentTimeMillis();
                Log.d(TAG, "Eyes detection : " + (te - ts) + " ms.");
                
                if (eyesArray != null) {
                	double leftX = (eyesArray[0].tl().x + eyesArray[0].br().x)/2;
                	double rightX = (eyesArray[1].tl().x + eyesArray[1].br().x)/2;
                	double noseX = (nose.tl().x + nose.br().x)/2;

                	
                	double d = leftX - rightX;
                	direction = (int)((leftX - noseX) * 150 / d) - 75 + "";
                    

    		        for (int j = 0; j < eyesArray.length; j++) {
    		        	
    		        	Mat eyeMat = faceMat.submat(eyesArray[j]);
    		        	
    		        	ts = System.currentTimeMillis();
    		        	Point pupil = Tracker.detectPupil(eyeMat);
    		        	te = System.currentTimeMillis();
    		            Log.d(TAG, "Pupil detection : " + (te - ts) + " ms.");
    		            if (pupil != null) {
    		            	Core.circle(mGrayClone, Tracker.offset(facesArray[i].tl(), Tracker.offset(eyesArray[j].tl(), pupil)), 3, PUPIL_COLOR, -1, 8, 0);
    		            }
    		                     	
    		            
    		            Log.d(TAG, "going to save image... " + savingImage);
    		            if (savingImage) {
    		            	Log.e(TAG, "saving image...");
    		            	MyUtils.saveImage(mGrayClone, "1 - color image");
    		            	MyUtils.saveImage(mGray, "2 - gray image");
    		                MyUtils.saveImage(mGray.submat(facesArray[i]), "3 - face");
    		                MyUtils.saveImage(faceMat, "4 - face and manual eye approximation");
    		                MyUtils.saveImage(eyeMat, "5 - eye");
    		                //MyUtils.saveImage(testMat, "6 - test");
    		                
    		                //MyUtils.saveImage(teye, "6 - thresholded eye");
    		                //MyUtils.saveImage(teyec, "7 - thresholded eye with contours");
    		                savingImage = false;
    		            }
    		
    		            
    		            Core.rectangle(mGrayClone, Tracker.offset(facesArray[i].tl(), eyesArray[j].tl()), Tracker.offset(facesArray[i].tl(), eyesArray[j].br()), EYES_RECT_COLOR, 3);
    		        }
                } else {
                	Log.d(TAG, "Haven't found an eye.");
                }
                
                Core.rectangle(mGrayClone, Tracker.offset(facesArray[i].tl(), nose.tl()), Tracker.offset(facesArray[i].tl(), nose.br()), EYES_RECT_COLOR, 3);
            }            
            else {
            	Log.d(TAG, "Haven't found any nose.");
            }            
            Core.rectangle(mGrayClone, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }
        	

        Bitmap bmp = Bitmap.createBitmap(mGrayClone.cols(), mGrayClone.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mGrayClone, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        long tee = System.currentTimeMillis();
        Log.d(TAG, "Processing frame : " + (tee - tss) + " ms.");
        return bmp;
        
    }
    
    private Point[] detectCorners(Mat face, Rect eye) {
    	Point tl = new Point(eye.x - 0.1 * eye.width, eye.y + 0.25 * eye.height);
    	Point br = new Point(eye.br().x + 0.1 * eye.width, eye.br().y - 0.25 * eye.height);
    	Rect newRoi = new Rect(tl, br);            	

    	Mat testMat = face.submat(newRoi);
    	
    	// We blur the center of the eyes to not be annoyed by pupil or whatever, but just need corners
    	Mat center = testMat.colRange(testMat.cols() / 4, 3 * testMat.cols() / 4);
    	Imgproc.GaussianBlur(center, center, new Size(9, 9), 2, 2);

    	MatOfPoint points = new MatOfPoint();
    	Imgproc.goodFeaturesToTrack(testMat, points, 2, 0.01, testMat.width() / 1.5);
    	return points.toArray();
    }
    
    
	private Rect[] detectFaces(Mat img) {
        MatOfRect faces = new MatOfRect();
        
        faceClassifier.detectMultiScale(img, faces, 1.2, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows()/3, img.cols()/3), new Size(img.rows(), img.cols()));
        
        return faces.toArray();
	}
	
	private Rect[] detectEyes(Mat face) {
		// Manual approximation of the eyes area
		Rect roi = new Rect(new Point(0, face.rows() / 5), new Point(face.cols(), face.rows() / 1.8));
        face = face.submat(roi);
        
        Mat rightArea = face.colRange(0, face.cols() / 2);
        Mat leftarea = face.colRange(face.cols() / 2, face.cols());

		MatOfRect rightEye = new MatOfRect();
		MatOfRect leftEye = new MatOfRect();

		eyeClassifier.detectMultiScale(rightArea, rightEye, 1.2, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(rightArea.cols() / 3, rightArea.cols() / 3), new Size(rightArea.rows(), rightArea.rows()));
		eyeClassifier.detectMultiScale(leftarea, leftEye, 1.2, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(leftEye.cols() / 3, leftEye.cols() / 3), new Size(leftEye.rows(), leftEye.rows()));

		if (rightEye.empty() || leftEye.empty()) {
			return null;
		}
		
		Rect[] rightArray = rightEye.toArray();
		Rect[] leftArray = leftEye.toArray();
		
		
		Rect[] eyesArray = new Rect[2];
		if (rightArray[0] != null && leftArray[0] != null) {
			eyesArray[0] = offset(leftArray[0], new Point(face.cols() / 2, roi.tl().y));
			eyesArray[1] = offset(rightArray[0], roi.tl());
		}
		
        return eyesArray;
	}
	
	public Rect detectNose(Mat face) {
		Rect roi = new Rect(new Point(0, face.rows() / 4), new Point(face.cols(), 3 * face.rows() / 4));
		face = face.submat(roi);
		
		MatOfRect noses = new MatOfRect();
		
		noseClassifier.detectMultiScale(face, noses, 1.2, 1, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(0, 0), new Size(face.cols(), face.rows()));

		if (noses.empty()) {
			return null;
		}
		
		return offset((noses.toArray())[0], roi.tl());
	}
	
	public static Point detectPupil(Mat eye) {
		Mat eyeClone = eye.clone();
		
		// Manual approximation of the pupil area
		Rect roi = new Rect(new Point(0, eyeClone.rows() / 3), new Point(eyeClone.cols(), 2 * eyeClone.rows() / 3));
		eyeClone = eyeClone.submat(roi);
		
        Imgproc.GaussianBlur(eyeClone, eyeClone, new Size(9, 9), 2, 2);
        
        double minBrightness = 255;
        
        for (int a = 0; a < eyeClone.rows(); a = a + 5) {
        	for (int b = 0; b < eyeClone.cols(); b = b + 5) {
        		double[] pixel = eyeClone.get(a, b);
        		
        		if (pixel[0] < minBrightness) {
        			minBrightness = pixel[0];
        		}
        	}
        }
        
        Imgproc.threshold(eyeClone, eyeClone, minBrightness + 5, 255, Imgproc.THRESH_BINARY_INV);
        
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();        
        Imgproc.findContours(eyeClone, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

		// Pupil should be the biggest blob. Let's find it.
		double pupilArea = 0;
		Point[] pupilContour = null;
		
        for (MatOfPoint contour : contours) {
        	double area = Imgproc.contourArea(contour);
        	
        	if (area > pupilArea) {
        		pupilArea = area;
        		pupilContour = contour.toArray();
        	}
		}
        
        Point bar = null;
        
        if (pupilContour != null) {
        	bar = new Point(0, 0);
            
            for (int p = 0; p < pupilContour.length; p++) {
            	bar.x += pupilContour[p].x;
            	bar.y += pupilContour[p].y;
            }
            
            bar.x /= pupilContour.length;
            bar.y /= pupilContour.length;

            
            offset(bar, roi.tl());
        }
        
        return bar;        
	}
	
	public static Point detectWhite(Mat eye) {
		Mat eyeClone = eye.clone();
		
		// Manual approximation of the eye itself
		Rect roi = new Rect(new Point(0, eyeClone.rows() / 3), new Point(eyeClone.cols(), 2 * eyeClone.rows() / 3));
		eyeClone = eyeClone.submat(roi);
		
        
        double minBrightness = 0;
        
        for (int a = 0; a < eyeClone.rows(); a = a + 5) {
        	for (int b = 0; b < eyeClone.cols(); b = b + 5) {
        		double[] pixel = eyeClone.get(a, b);
        		
        		if (pixel[0] > minBrightness) {
        			minBrightness = pixel[0];
        		}
        	}
        }
        
        Imgproc.threshold(eyeClone, eyeClone, minBrightness + 2, 255, Imgproc.THRESH_BINARY);
        
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();        
        Imgproc.findContours(eyeClone, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

		// Pupil should be the biggest blob. Let's find it.
        Point bar = new Point(0, 0);
        double numberOfPoints = 0;
		
        for (MatOfPoint contour : contours) {
        	Point[] points = contour.toArray();
        	numberOfPoints += points.length;
        	for (int p  = 0; p < points.length; p++) {
        		offset(bar, points[p]);
        	}
        }
        
        bar.x /= numberOfPoints;
        bar.y /= numberOfPoints;
        offset(bar, roi.tl());
        
        return bar;  
	
	}
	
	public static Point offset(Point p, Point offset) {
		p.x += offset.x;
		p.y += offset.y;
		return p;
    }
	
	public static Rect offset(Rect r, Point offset) {
		r.x += offset.x;
		r.y += offset.y;
		return r;
	}
	
	public void draw(Canvas canvas, float offsetx, float offsety) {
        canvas.drawText(direction, 20 + offsetx, 10 + 50 + offsety, paint);
    }
}
