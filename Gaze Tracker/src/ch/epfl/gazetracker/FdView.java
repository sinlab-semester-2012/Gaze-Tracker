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
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            bufferFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
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
        capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);        
        
        MatOfRect faces = new MatOfRect();
        
        int minSize = mRgba.rows() < mRgba.cols() ? mRgba.rows() : mRgba.cols();
        
    	if (faceCascade != null) {
            faceCascade.detectMultiScale(mGray, faces, 1.2, 10, 0, new Size(minSize/2, minSize/24), new Size(minSize, minSize)); //maxSize is not used...
    	}
    	        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            MatOfRect eyes = new MatOfRect();

            Mat sub = mGray.submat(facesArray[i]);
            
            //long ts = System.currentTimeMillis();
            sub = sub.rowRange(0, sub.rows() / 2); // approximating the eye's position, may be improved, but will need a shifting if done
            eyesCascade.detectMultiScale(sub, eyes, 1.1, 10, 0, new Size(50, 50), new Size(0, 0));
            //long te = System.currentTimeMillis();
            
            //Log.e(TAG, "detection takes : " + (te - ts));
            
            
            Rect[] eyesArray = eyes.toArray();
            for (int j = 0; j < eyesArray.length; j++) {
            	
            	Mat subSub = sub.submat(eyesArray[j]);
            	//Mat circles = new Mat();
            	
            	
                
                subSub.submat(0, subSub.rows() * 9 / 10, 0, subSub.cols() * 9 / 10);
                Imgproc.GaussianBlur(subSub, subSub, new Size(9, 9), 2, 2 );

            	double brightness = 255;
            	int xpos = 0;
            	int ypos = 0;
                for (int a = 0; a < subSub.rows(); a = a + 5) {
                	for (int b = 0; b < subSub.cols(); b = b + 5) {
                		double[] pixel = subSub.get(a, b);
                		if (pixel[0] < brightness) {
                			brightness = pixel[0];
                			xpos = a;
                			ypos = b;
                		}
                	}
                }
                
                Core.circle(mRgba, sum(sum(eyesArray[j].tl(), facesArray[i].tl()),  new Point(xpos,  ypos)), 4, CIRCLES_COLOR);
                Mat teye = subSub.clone();
                Imgproc.threshold(subSub, teye, brightness + 5, 255, Imgproc.THRESH_BINARY);
                
                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                Mat h = new Mat();
                Mat teyec = teye.clone();
                Imgproc.findContours(teyec.clone(), contours, h, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
                Log.e(TAG, "number of contours found : " + contours.size());
                
                if (contours.size() == 2) {
                	Point[] contoursP = contours.get(1).toArray();
                	Log.e(TAG, "number of points " + contoursP.length);
            		Point bar = new Point(0, 0);

                	for (int p = 0; p < contoursP.length; p++) {
                		bar = sum(bar, contoursP[p]);
                	}
                	
                	
                	bar = new Point(bar.x / contoursP.length, bar.y / contoursP.length);
                    Core.circle(teyec, bar, 3, BLACK_COLOR, -1, 8, 0 );

                }
                //Imgproc.cvtColor(teyec, teyec, Imgproc.COLOR_RGB2RGBA);
                
                //Imgproc.drawContours(teyec, contours, -1, CIRCLES_COLOR, -1);
                
                
                Log.e(TAG, "going to save image... " + savingImage);
                if (savingImage) {
                	Log.e(TAG, "saving image...");
                	MyUtils.saveImage(mRgba, "1 - color image");
                	MyUtils.saveImage(mGray, "2 - gray image");
                    MyUtils.saveImage(mGray.submat(facesArray[i]), "3 - face");
                    MyUtils.saveImage(sub, "4 - face and manual eye approximation");
                    MyUtils.saveImage(subSub, "5 - eye");
                    MyUtils.saveImage(teye, "6 - thresholded eye");
                    MyUtils.saveImage(teyec, "7 - thresholded eye with contours");
                    Log.e(TAG, "saving image... done");
                    savingImage = false;

                }
                
            	//Imgproc.threshold(subSub, subSub, 35, 255, Imgproc.THRESH_BINARY);
            	//Imgproc.HoughCircles(subSub, circles, Imgproc.CV_HOUGH_GRADIENT, 1, 1, 200, 100, 0, 0);            	
            	
                /*
            	if (circles.cols() == 0) {
            		Log.e(TAG, "no pupils detected :(");
            	} else {
            		Log.e(TAG, "pupils detected !");
            	}
            	
            	for (int x = 0; x < circles.cols(); x++) 
                {
                        double vCircle[] = circles.get(0,x);

                        Point center = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
                        int radius = (int)Math.round(vCircle[2]);
                        // draw the circle center
                        Core.circle(mRgba, center, 3, CIRCLES_COLOR, -1, 8, 0 );
                        // draw the circle outline
                        Core.circle(mRgba, center, radius, CIRCLES_COLOR, 3, 8, 0 );

                }
				*/
            	
            	
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
