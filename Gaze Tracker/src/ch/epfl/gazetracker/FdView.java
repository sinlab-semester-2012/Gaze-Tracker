package ch.epfl.gazetracker;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import ch.epfl.gazetracker.R;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class FdView extends SampleCvViewBase {
    private static final String   TAG = "Sample::FdView";
    
    private Mat                   mRgba;
    private Mat                   mGray;
    private File                  bufferFile;
    private CascadeClassifier 	  facesClassifier;
    private CascadeClassifier     eyesClassifier;
    
    private boolean				  savingImage = true;

    private static final Scalar   FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(255, 0, 0, 255);
    private static final Scalar   CIRCLES_COLOR = new Scalar(0, 0, 255, 255);
    
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
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }

    @Override
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
        }

        super.surfaceCreated(holder);
	}
    
    private Point sum(Point a, Point b) {
    	return new Point(a.x + b.x, a.y + b.y);
    }

	@Override
    protected Bitmap processFrame(VideoCapture capture) {
        long ts = System.currentTimeMillis();
        capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);        
        long te = System.currentTimeMillis();            
        Log.d(TAG, "Retrieving pictures from camera : " + (te - ts) + " ms.");
        
        ts = System.currentTimeMillis();
        Rect[] facesArray = Tracker.detectFaces(mGray, facesClassifier);        
        te = System.currentTimeMillis();            
        Log.d(TAG, "Face detection : " + (te - ts) + " ms.");
        Log.d(TAG, facesArray.length + " faces detected.");
        
        for (int i = 0; i < facesArray.length; i++) {
            Mat faceMat = mGray.submat(facesArray[i]);
            Rect roi = new Rect(new Point(0, faceMat.rows() / 5), new Point(faceMat.cols(), faceMat.rows() / 2));
            faceMat = faceMat.submat(roi);
            
            ts = System.currentTimeMillis();
            Rect[] eyesArray = Tracker.detectEyes(faceMat, eyesClassifier);
            te = System.currentTimeMillis();
            Log.d(TAG, "Eyes detection : " + (te - ts) + " ms.");
            
            for (int j = 0; j < eyesArray.length; j++) {
            	Mat eyeMat = faceMat.submat(eyesArray[j]);
                
                ts = System.currentTimeMillis();
            	Point pupil = Tracker.detectPupil(eyeMat);
            	te = System.currentTimeMillis();
                Log.d(TAG, "Pupil detection : " + (te - ts) + " ms.");
                
                Log.d(TAG, "going to save image... " + savingImage);
                if (savingImage) {
                	Log.e(TAG, "saving image...");
                	MyUtils.saveImage(mRgba, "1 - color image");
                	MyUtils.saveImage(mGray, "2 - gray image");
                    MyUtils.saveImage(mGray.submat(facesArray[i]), "3 - face");
                    MyUtils.saveImage(faceMat, "4 - face and manual eye approximation");
                    MyUtils.saveImage(eyeMat, "5 - eye");
                    //MyUtils.saveImage(teye, "6 - thresholded eye");
                    //MyUtils.saveImage(teyec, "7 - thresholded eye with contours");
                    savingImage = false;
                }

                if (pupil != null) {
                    Core.circle(mRgba, sum(facesArray[i].tl(), sum(roi.tl(), sum(eyesArray[j].tl(),  pupil))), 3, CIRCLES_COLOR, -1, 8, 0);
                }
                
                Core.rectangle(mRgba, sum(facesArray[i].tl(), sum(roi.tl(), eyesArray[j].tl())), sum(facesArray[i].tl(), sum(roi.tl(), eyesArray[j].br())), EYES_RECT_COLOR, 3);
            }
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }
        	

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mRgba, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        return bmp;
        
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (bufferFile != null)
            	bufferFile.delete();

            mRgba = null;
            mGray = null;
            bufferFile = null;
        }
    }
}
