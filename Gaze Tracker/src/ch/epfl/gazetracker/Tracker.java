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
    private static final String TAG = "Tracker";
    private static final Paint PAINT = new Paint();
    private static final Scalar   FACE_RECT_COLOR = new Scalar(0);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(51);
    private static final Scalar   PUPIL_COLOR = new Scalar(255);
    
    private Mat mGray;
    
    private String direction;
    private String directionEye;
    private String totalDir;
    
    private boolean debugImage = false;
	
	private CascadeClassifier 	  faceClassifier;
    private CascadeClassifier     eyeClassifier;
    private CascadeClassifier     noseClassifier;
    
    public Tracker(Context context) {
    	faceClassifier = new CascadeClassifier();
    	loadClassifier(context, faceClassifier, R.raw.haarcascade_frontalface_alt2, "haarcascade_frontalface_alt2.xml");
    	eyeClassifier = new CascadeClassifier();
    	loadClassifier(context, eyeClassifier, R.raw.haarcascade_eye, "haarcascade_eye.xml");
    	noseClassifier = new CascadeClassifier();
    	loadClassifier(context, noseClassifier, R.raw.haarcascade_mcs_nose, "haarcascade_mcs_nose.xml");
    	
    	direction = "";
    	directionEye = "";
    	totalDir = "";
    	
    	mGray = new Mat();
    	
        PAINT.setColor(Color.YELLOW);
        PAINT.setTextSize(50);
    }
    
    protected Bitmap processFrame(VideoCapture capture) {
		long tss = System.currentTimeMillis();
		
        long ts = System.currentTimeMillis();
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
        long te = System.currentTimeMillis();
        Log.d(TAG, "Retrieving pictures from camera : " + (te - ts) + " ms.");
        
        direction = "Face(s) not found.";
        directionEye = "Face(s) not found.";
    	totalDir = "Face(s) not found.";
        
        ts = System.currentTimeMillis();
        Rect[] facesArray = detectFaces(mGray);        
        te = System.currentTimeMillis();            
        Log.d(TAG, "Face detection : " + (te - ts) + " ms.");
        
        for (int i = 0; i < facesArray.length; i++) {
        	long tfs = System.currentTimeMillis();
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
                	ts = System.currentTimeMillis();
                	Point[] leftCorners = detectCorners(faceMat, eyesArray[0], true);
                	Point[] rightCorners = detectCorners(faceMat, eyesArray[1], false);
                    te = System.currentTimeMillis();
                    Log.d(TAG, "Corners detection : " + (te - ts) + " ms.");
                	
                	if (leftCorners != null && rightCorners != null) {
                    	Point leftPupil = detectPupil(faceMat.submat(eyesArray[0]));
                    	
                    	if (leftPupil != null) {
                    		double leftX = (eyesArray[0].tl().x + eyesArray[0].br().x)/2;
                        	double rightX = (eyesArray[1].tl().x + eyesArray[1].br().x)/2;
                        	double noseX = (nose.tl().x + nose.br().x)/2;

                        	double d = leftX - rightX;
                        	direction = "Eyes: " + (int)(((leftX - noseX) * 150 / d) - 75);
                        	
                        	double eyeLength = leftCorners[1].x - leftCorners[0].x;
                        	Log.d(TAG, "eyeLength : " + eyeLength);
                        	double dprime = leftCorners[1].x - (leftPupil.x + faceMat.submat(eyesArray[0]).cols()/10);
                        	Log.d(TAG, "dprime : " + dprime);

                        	directionEye = "Face : " + (int)((360 / eyeLength) * dprime - 180);
                        	
                        	totalDir = "Total: " + ((int)((leftX - noseX) * 150 / d) - 75 + (int)((360 / eyeLength) * dprime - 180)) + "";
                        	
                        	Core.circle(mGray, Tracker.offset(facesArray[i].tl(), Tracker.offset(eyesArray[0].tl(), leftPupil)), 3, PUPIL_COLOR, -1, 8, 0);
                    	} else {
                        	direction = "Pupil not found.";
                        	directionEye = "Pupil not found.";
                        	totalDir = "Pupil not found.";
                        	Log.d(TAG, "Haven't found a pupil.");
                        }
                		
                	} else {
                    	direction = "Corners not found.";
                    	directionEye = "Corners not found.";
                    	totalDir = "Corners not found.";
                    	Log.d(TAG, "Haven't found corners.");
                    }
                	
                	Log.d(TAG, "going to save image... " + debugImage);
		            if (debugImage) {
		            	Log.e(TAG, "saving image...");
		            	MyUtils.saveImage(mGray, "1 - gray image");
		                MyUtils.saveImage(mGray.submat(facesArray[i]), "2 - face");
		                debugImage = false;
		            }
		            
		            Core.rectangle(mGray, Tracker.offset(facesArray[i].tl(), eyesArray[0].tl()), Tracker.offset(facesArray[i].tl(), eyesArray[0].br()), EYES_RECT_COLOR, 3);
		            Core.rectangle(mGray, Tracker.offset(facesArray[i].tl(), eyesArray[1].tl()), Tracker.offset(facesArray[i].tl(), eyesArray[1].br()), EYES_RECT_COLOR, 3);
                } else {
                	direction = "Eye(s) not found.";
                	directionEye = "Eye(s) not found.";
                	totalDir = "Eye(s) not found.";
                	Log.d(TAG, "Haven't found an eye.");
                }
                
                Core.rectangle(mGray, Tracker.offset(facesArray[i].tl(), nose.tl()), Tracker.offset(facesArray[i].tl(), nose.br()), EYES_RECT_COLOR, 3);
            } else {
            	direction = "Nose not found.";
            	directionEye = "Nose not found.";
            	totalDir = "Nose not found.";
            	Log.d(TAG, "Haven't found any nose.");
            }
            Core.rectangle(mGray, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            
            long tfe = System.currentTimeMillis();            
            Log.d(TAG, "Face processing : " + (tfe - tfs) + " ms.");
        }
        	

        Bitmap bmp = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mGray, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        long tee = System.currentTimeMillis();
        Log.d(TAG, "Processing frame : " + (tee - tss) + " ms.");
        return bmp;
    }

    /**
     * Detects faces of different sizes in the input image. The detected faces are returned as an array of rectangles.
     * @param img Matrix containing an image where faces need to be detected.
     * @return An array of rectangles where each rectangle contains the detected face.
     * @see <a href="http://www.google.fr/">http://www.google.fr/</a>
     */
	private Rect[] detectFaces(Mat img) {
        MatOfRect faces = new MatOfRect();
        
       	//The minimum size is set to be a third of the image size.
        faceClassifier.detectMultiScale(img, faces, 1.2, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows()/3, img.cols()/3), new Size(img.rows(), img.cols()));
        
        return faces.toArray();
	}

	/**
     * Detects one nose in the input image. The detected nose is returned as a rectangle.
     * @param img Matrix containing a face where nose need to be detected.
     * @return A rectangle containing the detected nose, or null if no nose has been detected.
     */
	public Rect detectNose(Mat face) {
		Rect roi = new Rect(new Point(face.cols() / 4, face.rows() / 4), new Point(3 * face.cols() / 4, 3 * face.rows() / 4));
		face = face.submat(roi);
		
		MatOfRect noses = new MatOfRect();
		
		noseClassifier.detectMultiScale(face, noses, 1.2, 1, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(face.rows() / 2, face.rows() / 2), new Size(face.cols(), face.rows()));

		if (noses.empty()) {
			return null;
		}
		
		return offset((noses.toArray())[0], roi.tl());
	}
    
    
    private Point[] detectCorners(Mat face, Rect eye, boolean left) {
    	Point tl = new Point(eye.x - 0.1 * eye.width, eye.y + 0.25 * eye.height);
    	Point br = new Point(eye.br().x + 0.1 * eye.width, eye.br().y - 0.25 * eye.height);
    	Rect newRoi = new Rect(tl, br);

    	Mat testMat = face.submat(newRoi);
    	
    	Mat leftCorner = testMat.colRange(0, testMat.cols() / 3);
    	Mat rightCorner = testMat.colRange(2 * testMat.cols() / 3, testMat.cols());
    	
    	if (debugImage) {
    		MyUtils.saveImage(testMat, "AreaEye");
    		MyUtils.saveImage(leftCorner, "leftAreaEye");
    		MyUtils.saveImage(rightCorner, "rightAreaEye");   
    		//debugImage = false;
    	}
    	
    	
    	MatOfPoint leftPoints = new MatOfPoint();
    	Imgproc.goodFeaturesToTrack(leftCorner, leftPoints, 1, 0.01, 0, new Mat(), 15, true, 0.04);
    	MatOfPoint rightPoints = new MatOfPoint();
    	Imgproc.goodFeaturesToTrack(rightCorner, rightPoints, 1, 0.01, 0, new Mat(), 15, true, 0.04);
    	
    	if (leftPoints.empty() || rightPoints.empty()) {
    		return null;
    	}
    	
    	Point[] leftArray = leftPoints.toArray();
    	Point[] rightArrPoints = rightPoints.toArray();
    	
    	Core.circle(testMat, leftArray[0], 3, PUPIL_COLOR, -1, 8, 0);
    	Core.circle(testMat, offset(rightArrPoints[0], new Point(4 * testMat.cols() / 5, 0)), 3, PUPIL_COLOR, -1, 8, 0);
    	if (debugImage) {
        	
        	MyUtils.saveImage(testMat, "testCorners");
    	}
    	
    	Point[] cornersArray = new Point[2];
    	if (left) {
    		cornersArray[0] = leftArray[0];
    		cornersArray[1] = rightArrPoints[0];
    	} else {
    		cornersArray[0] = rightArrPoints[0];
    		cornersArray[1] = leftArray[0];
    	}
    	
    	return cornersArray;
    }
    
    
	
	private Rect[] detectEyes(Mat face) {
		// Manual approximation of the eyes area
		Rect roi = new Rect(new Point(0, face.rows() / 5), new Point(face.cols(), face.rows() / 2));
        face = face.submat(roi);
        
        Mat rightArea = face.colRange(0, face.cols() / 2);
        Mat leftarea = face.colRange(face.cols() / 2, face.cols());

		MatOfRect rightEye = new MatOfRect();
		MatOfRect leftEye = new MatOfRect();

		eyeClassifier.detectMultiScale(rightArea, rightEye, 1.2, 1, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(rightArea.cols() / 3, rightArea.cols() / 3), new Size(rightArea.rows(), rightArea.rows()));
		eyeClassifier.detectMultiScale(leftarea, leftEye, 1.2, 1, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(leftEye.cols() / 3, leftEye.cols() / 3), new Size(leftEye.rows(), leftEye.rows()));

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
	
	
	/**
	 * Detects the pupil in the input image. The pupil position is returned as a point.
	 * @param eye Matrix containing an eye where the pupil need to be detected.
	 * @return The pupil center position.
	 */
	public static Point detectPupil(Mat eye) {
		//We clone the matrix, to not break the original image.
		Mat eyeClone = eye.clone();
		
		//Manual approximation of the pupil area.
		Rect roi = new Rect(new Point(0, eyeClone.rows() / 3), new Point(eyeClone.cols(), 2 * eyeClone.rows() / 3));
		eyeClone = eyeClone.submat(roi);
		
        Imgproc.GaussianBlur(eyeClone, eyeClone, new Size(9, 9), 2, 2);
        
        //We now have to look for <i>one of<i> the darkest pixel, which will belong to the pupil.
        double minBrightness = 255;
        
        for (int a = 0; a < eyeClone.rows(); a = a + 5) {
        	for (int b = 0; b < eyeClone.cols(); b = b + 5) {
        		double[] pixel = eyeClone.get(a, b);
        		
        		if (pixel[0] < minBrightness) {
        			minBrightness = pixel[0];
        		}
        	}
        }
        
        //When it's done, we threshold the image to only keep the darkest area.
        Imgproc.threshold(eyeClone, eyeClone, minBrightness + 5, 255, Imgproc.THRESH_BINARY_INV);
        
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();        
        Imgproc.findContours(eyeClone, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        
		//Pupil should be the biggest blob. Let's find it.
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
	
	private void loadClassifier(Context context, CascadeClassifier classifier, int id, String name) {
		try {
			InputStream is = context.getResources().openRawResource(id);
	        File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
	        File bufferFile = new File(cascadeDir, name);
	        FileOutputStream os = new FileOutputStream(bufferFile);
	        byte[] buffer = new byte[4096];
	        int bytesRead;
	        while ((bytesRead = is.read(buffer)) != -1) {
	            os.write(buffer, 0, bytesRead);
	        }
	        is.close();
	        os.close();
	        classifier.load(bufferFile.getAbsolutePath());
	        if (classifier.empty()) {
	            Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
	            faceClassifier = null;
	        } else {
	            Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
	        }
            cascadeDir.delete();            
		} catch (IOException e) {
			 e.printStackTrace();
	         Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
		}
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
        canvas.drawText(direction, 10 + offsetx, 10 + 50 + offsety, PAINT);
        canvas.drawText(directionEye, 10 + offsetx, 10 + 50 + 50 + offsety, PAINT);
        canvas.drawText(totalDir, 10 + offsetx, 10 + 50 + 50 + 50 + offsety, PAINT);
    }
}
