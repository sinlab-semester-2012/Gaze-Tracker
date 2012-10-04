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
    private CascadeClassifier 	  faceCascade;
    private CascadeClassifier     eyesCascade;
    
    private boolean				  savingImage = true;

    private static final Scalar   FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(255, 0, 0, 255);
    private static final Scalar   CIRCLES_COLOR = new Scalar(0, 0, 255, 255);
    private static final Scalar   BLACK_COLOR = new Scalar(255, 255, 255, 255);
    
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

            faceCascade = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (faceCascade.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                faceCascade = null;
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

            eyesCascade = new CascadeClassifier(bufferFile.getAbsolutePath());

            if (eyesCascade.empty()) {
                Log.e(TAG, "Failed to load cascade classifier : " + bufferFile.getAbsolutePath());
                eyesCascade = null;
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
        Log.e(TAG, "retrieving pictures from camera takes : " + (te - ts));
        
        MatOfRect faces = new MatOfRect();
        
        int minSize = mRgba.rows() < mRgba.cols() ? mRgba.rows() : mRgba.cols();
        
        ts = System.currentTimeMillis();
        faceCascade.detectMultiScale(mGray, faces, 1.2, 8, Objdetect.CASCADE_SCALE_IMAGE, new Size(minSize/2, minSize/2), new Size(minSize, minSize));
        te = System.currentTimeMillis();            
        Log.e(TAG, "face detection takes : " + (te - ts));
        
        Rect[] facesArray = faces.toArray();        
        Log.e(TAG, "number of faces detected : " + facesArray.length);
        
        for (int i = 0; i < facesArray.length; i++) {
            MatOfRect eyes = new MatOfRect();
            Mat faceMat = mGray.submat(facesArray[i]);
            
            ts = System.currentTimeMillis();
            //approximating the eye's position, may be improved, but will need a shifting if done
            faceMat = faceMat.rowRange(0, faceMat.rows() / 2); 
            eyesCascade.detectMultiScale(faceMat, eyes, 1.2, 5, 0, new Size(faceMat.cols() / 6, faceMat.cols() / 6), new Size(faceMat.rows() / 4, faceMat.rows() / 4));
            te = System.currentTimeMillis();            
            Log.e(TAG, "eyes detection takes : " + (te - ts));

            Rect[] eyesArray = eyes.toArray();
            for (int j = 0; j < eyesArray.length; j++) {            	
            	Mat eyeMat = faceMat.submat(eyesArray[j]);
                
                ts = System.currentTimeMillis();
                Imgproc.GaussianBlur(eyeMat, eyeMat, new Size(9, 9), 2, 2);
                //Here we find one of the darkest pixel, one of because we don't check every pixel, but only 1/5
            	double brightness = 255;
            	//int xpos = 0;
            	//int ypos = 0;
                for (int a = 0; a < eyeMat.rows(); a = a + 5) {
                	for (int b = 0; b < eyeMat.cols(); b = b + 5) {
                		double[] pixel = eyeMat.get(a, b);
                		if (pixel[0] < brightness) {
                			brightness = pixel[0];
                			//xpos = a;
                			//ypos = b;
                		}
                	}
                }
                te = System.currentTimeMillis();            
                Log.e(TAG, "finding darkest pixel takes : " + (te - ts));
                
                Mat teye = eyeMat.clone();
                Imgproc.threshold(eyeMat, teye, brightness + 5, 255, Imgproc.THRESH_BINARY);
                
                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                Mat hierarchy = new Mat();
                
                //Need to find something better than this shit, because it detects the "outest" contour of the image i.e. the border -_-
                Imgproc.findContours(teye.clone(), contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
                Log.e(TAG, "number of contours found : " + contours.size());
        		Point bar = new Point(0, 0);

        		double maxArea = 0;
        		Point[] maxContour = null;
        		double pupilArea = 0;
        		Point[] pupilContour = null;
                for (MatOfPoint contour : contours) {
                	double area = Imgproc.contourArea(contour);
                	
                	if (area >= maxArea) {
                		pupilArea = maxArea;
                		pupilContour = maxContour;
                		maxArea = area;
                		maxContour = contour.toArray();
                	} else if (area >= pupilArea) {
                		pupilArea = area;
                		pupilContour = contour.toArray();
                	}
				}
                if (pupilContour != null) {
                	Log.e(TAG, "number of points " + pupilContour.length);
                	
                	for (int p = 0; p < pupilContour.length; p++) {
                		bar = sum(bar, pupilContour[p]);
                	}
                    
                	bar = new Point(bar.x / pupilContour.length, bar.y / pupilContour.length);

                    Core.circle(mRgba, sum(sum(eyesArray[j].tl(), facesArray[i].tl()),  bar), 3, CIRCLES_COLOR, -1, 8, 0);
                }
            	
                
                Log.e(TAG, "going to save image... " + savingImage);
                if (savingImage) {
                	Log.e(TAG, "saving image...");
                	MyUtils.saveImage(mRgba, "1 - color image");
                	MyUtils.saveImage(mGray, "2 - gray image");
                    MyUtils.saveImage(mGray.submat(facesArray[i]), "3 - face");
                    MyUtils.saveImage(faceMat, "4 - face and manual eye approximation");
                    MyUtils.saveImage(eyeMat, "5 - eye");
                    MyUtils.saveImage(teye, "6 - thresholded eye");
                    //MyUtils.saveImage(teyec, "7 - thresholded eye with contours");
                    Log.e(TAG, "saving image... done");
                    savingImage = false;
                }            	
            	
                Core.rectangle(mRgba, sum(eyesArray[j].tl(), facesArray[i].tl()), sum(eyesArray[j].br(), facesArray[i].tl()), EYES_RECT_COLOR, 3);
                Log.e(TAG, "" + eyesArray[j].size());                
            }
        	
        	
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            Log.e(TAG, "the size of the rectangle is : " + facesArray[i].height + " x " + facesArray[i].width);
            
            
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
