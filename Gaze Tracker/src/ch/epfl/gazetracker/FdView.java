package ch.epfl.gazetracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import ch.epfl.gazetracker.R;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.Log;
import android.view.SurfaceHolder;

class FdView extends SampleCvViewBase {
    private static final String   TAG = "Sample::FdView";
    
    private Mat                   mGrayClone;
    private Mat                   mGray;
    private File                  bufferFile;
    private CascadeClassifier 	  facesClassifier;
    private CascadeClassifier     eyesClassifier;
    private CascadeClassifier	  mouthClassifier;
    private CascadeClassifier     noseClassifier;
    
    private boolean				  savingImage = true;

    private static final Scalar   FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(255, 0, 0, 255);
    private static final Scalar   BLUE_COLOR = new Scalar(0, 0, 255, 255);
    private static final Scalar   WHITE_COLOR = new Scalar(255, 255, 255, 255);
    
    
    public FdView(Context context) {
    	    	
        super(context);

        try {
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            bufferFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
            FileOutputStream os = new FileOutputStream(bufferFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            facesClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (facesClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                facesClassifier = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            
            cascadeDir.delete();
            
            is = context.getResources().openRawResource(R.raw.haarcascade_eye);
            cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            bufferFile = new File(cascadeDir, "haarcascade_eye.xml");
            os = new FileOutputStream(bufferFile);

            buffer = new byte[4096];
            bytesRead = 0;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            eyesClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (eyesClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyesClassifier = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            
            cascadeDir.delete();
            
            is = context.getResources().openRawResource(R.raw.haarcascade_mcs_mouth);
            cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            bufferFile = new File(cascadeDir, "haarcascade_mcs_mouth.xml");
            os = new FileOutputStream(bufferFile);

            buffer = new byte[4096];
            bytesRead = 0;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mouthClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (eyesClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyesClassifier = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            
            cascadeDir.delete();
            
            is = context.getResources().openRawResource(R.raw.haarcascade_mcs_mouth);
            cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            bufferFile = new File(cascadeDir, "haarcascade_mcs_mouth.xml");
            os = new FileOutputStream(bufferFile);

            buffer = new byte[4096];
            bytesRead = 0;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mouthClassifier = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (eyesClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyesClassifier = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            
            cascadeDir.delete();
            
            is = context.getResources().openRawResource(R.raw.haarcascade_mcs_nose);
            cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
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

            if (eyesClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyesClassifier = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + bufferFile.getAbsolutePath());
            
            cascadeDir.delete();
            
            
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }

    @Override
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            mGray = new Mat();
            mGrayClone = new Mat();
        }

        super.surfaceCreated(holder);
	}

	@Override
    protected Bitmap processFrame(VideoCapture capture) {
		long tss = System.currentTimeMillis();
        long ts = System.currentTimeMillis();
        //capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);      
        mGrayClone = mGray.clone();
        long te = System.currentTimeMillis();            
        Log.d(TAG, "Retrieving pictures from camera : " + (te - ts) + " ms.");
        
        
        
        
        ts = System.currentTimeMillis();
        Rect[] facesArray = Tracker.detectFaces(mGray, facesClassifier);        
        te = System.currentTimeMillis();            
        Log.d(TAG, "Face detection : " + (te - ts) + " ms.");
        Log.d(TAG, facesArray.length + " faces detected.");
        
        for (int i = 0; i < facesArray.length; i++) {
            Mat faceMat = mGray.submat(facesArray[i]);
            
            ts = System.currentTimeMillis();
            Rect[] eyesArray = Tracker.detectEyes(faceMat, eyesClassifier);
            te = System.currentTimeMillis();
            Log.d(TAG, "Eyes detection : " + (te - ts) + " ms.");
            
            ///////////////////
            Rect nose = Tracker.detectNose(faceMat, noseClassifier);
            if (nose != null) {
                Core.rectangle(mGrayClone, Tracker.offset(facesArray[i].tl(), nose.tl()), Tracker.offset(facesArray[i].tl(), nose.br()), EYES_RECT_COLOR, 3);
            }
            
            //////////////////////////
            
            if (eyesArray != null && nose != null) {
            	Point centerLeftEye = Tracker.offset(eyesArray[0].tl(), eyesArray[0].br());
            	centerLeftEye.x /= 2;
            	centerLeftEye.y /= 2;
            	
            	Point centerRightEye = Tracker.offset(eyesArray[1].tl(), eyesArray[1].br());
            	centerRightEye.x /= 2;
            	centerRightEye.y /= 2;
            	
            	Point centerNose = Tracker.offset(nose.tl(), nose.br());
            	centerNose.x /= 2;
            	centerNose.y /= 2;
            	
            	double d = centerLeftEye.x - centerRightEye.x;
            	String direction = "";
            	if (centerLeftEye.x - centerNose.x < d/7) {
            		direction = "45°L";
            	} else if (centerLeftEye.x - centerNose.x < 2*d/7){
            		direction = "30°L";
            	} else if (centerLeftEye.x - centerNose.x < 3*d/7){
            		direction = "15°L";
            	} else if (centerLeftEye.x - centerNose.x < 4*d/7){
            		direction = "0°";
            	} else if (centerLeftEye.x - centerNose.x < 5*d/7){
            		direction = "15°R";
            	} else if (centerLeftEye.x - centerNose.x < 6*d/7){
            		direction = "30°R";
            	} else if (centerLeftEye.x - centerNose.x < d){
            		direction = "45°R";
            	} else {
            		direction = "error";
            	}
            	
            	Log.e(TAG, "The direction is : " + direction);
            }
            
            
            
            if (eyesArray != null) {
            
            for (int j = 0; j < eyesArray.length; j++) {
            	
            	Mat eyeMat = faceMat.submat(eyesArray[j]);
            	
            	ts = System.currentTimeMillis();
            	Point pupil = Tracker.detectPupil(eyeMat);
            	te = System.currentTimeMillis();
                Log.d(TAG, "Pupil detection : " + (te - ts) + " ms.");
                if (pupil != null) {
                	Core.circle(mGrayClone, Tracker.offset(facesArray[i].tl(), Tracker.offset(eyesArray[j].tl(), pupil)), 3, WHITE_COLOR, -1, 8, 0);
                }
                /*
                ts = System.currentTimeMillis();
            	Point white = Tracker.detectWhite(eyeMat);
            	te = System.currentTimeMillis();
                Log.d(TAG, "White detection : " + (te - ts) + " ms.");
                if (pupil != null) {
                	Core.circle(mGrayClone, Tracker.offset(facesArray[i].tl(), Tracker.offset(eyesArray[j].tl(), white)), 3, WHITE_COLOR, -1, 8, 0);
                }
                
                Rect roi = new Rect(new Point(0, eyeMat.rows() / 3), new Point(eyeMat.cols(), 2 * eyeMat.rows() / 3));
                Mat e2 = eyeMat.clone();
                e2 = e2.submat(roi);
                Imgproc.Canny(e2, e2, 25, 75);
                if (savingImage) {
                    MyUtils.saveImage(e2, "myTest");

                }
                */
            	/* Method with the line linking the eyes' corners
            	Point tl = new Point(eyesArray[j].x - 0.1 * eyesArray[j].width, eyesArray[j].y + 0.25 * eyesArray[j].height);
            	Point br = new Point(eyesArray[j].br().x + 0.1 * eyesArray[j].width, eyesArray[j].br().y - 0.25 * eyesArray[j].height);
            	Rect newRoi = new Rect(tl, br);            	

            	Mat testMat = faceMat.submat(newRoi);
            	
            	// We blur the center of the eyes to not be annoyed by pupil or whatever, but just need corners
            	Mat center = testMat.colRange(testMat.cols() / 4, 3 * testMat.cols() / 4);
            	Imgproc.GaussianBlur(center, center, new Size(9, 9), 2, 2);

            	MatOfPoint points = new MatOfPoint();
            	Imgproc.goodFeaturesToTrack(testMat, points, 2, 0.01, testMat.width() / 1.5);
            	
            	Point[] pointsArrPoints = points.toArray();
            	
            	//for (int p = 0; p < pointsArrPoints.length; p++) {
            	//	//Core.line(img, pt1, pt2, color)
                //    Core.circle(testMat, pointsArrPoints[p], 3, WHITE_COLOR, -1, 8, 0);
                //    Core.circle(mRgba, sum(facesArray[i].tl(), sum(roi.tl(), sum(eyesArray[j].tl(),  pointsArrPoints[p]))), 3, WHITE_COLOR, -1, 8, 0);
            	//}
            	
            	
            	if (pointsArrPoints.length == 2) {
            		Point p1 = sum(facesArray[i].tl(), sum(newRoi.tl(),  pointsArrPoints[0]));
                	Point p2 = sum(facesArray[i].tl(), sum(newRoi.tl(),  pointsArrPoints[1]));
                	Core.line(mRgba, p1, p2, WHITE_COLOR);

            	}

            	*/
            	
            	
                
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
	
    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mGrayClone != null)
                mGrayClone.release();
            if (mGray != null)
                mGray.release();
            if (bufferFile != null)
            	bufferFile.delete();

            mGrayClone = null;
            mGray = null;
            bufferFile = null;
        }
    }
}
