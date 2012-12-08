package ch.epfl.gazetracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import netP5.NetAddress;

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

import oscP5.OscMessage;
import oscP5.OscP5;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

public class Tracker {
    private static final String TAG = "Tracker";
    private static final String id = UUID.randomUUID().toString();

	private NetAddress myBroadcastLocation;
	
    private static final Paint 	  PAINT = new Paint();
    private static final Scalar   FACE_RECT_COLOR = new Scalar(0);
    private static final Scalar	  EYES_RECT_COLOR = new Scalar(51);
    private static final Scalar   PUPIL_COLOR = new Scalar(255);

    private Mat mGray;

    private String direction;
    private String directionEye;
    private String totalDir;

    private boolean debugImage = true;

	private CascadeClassifier 	  faceClassifier;
    private CascadeClassifier     eyeClassifier;
    private CascadeClassifier     noseClassifier;

    public Tracker(Context context) {
    	faceClassifier = new CascadeClassifier();
    	loadClassifier(context, faceClassifier, R.raw.haarcascade_frontalface_alt2, "haarcascade_frontalface_alt2.xml");
    	eyeClassifier = new CascadeClassifier();
    	loadClassifier(context, eyeClassifier, R.raw.haarcascade_eye, "haarcascade_eye");
    	noseClassifier = new CascadeClassifier();
    	loadClassifier(context, noseClassifier, R.raw.haarcascade_mcs_nose, "haarcascade_mcs_nose.xml");

    	direction = "";
    	directionEye = "";
    	totalDir = "";

    	mGray = new Mat();

        PAINT.setColor(Color.YELLOW);
        PAINT.setTextSize(50);

        myBroadcastLocation = new NetAddress("128.179.154.237",32000);
    }

    protected Bitmap processFrame(VideoCapture capture) {
		long processingStart = System.currentTimeMillis();
		
        capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
        
        long ts = System.currentTimeMillis();
        Rect face = detectFaces(mGray);
        long te = System.currentTimeMillis();
        Log.d(TAG, "Face detection : " + (te - ts) + " ms.");
        
        if (face != null) {
            Mat faceMat = mGray.submat(face);
            
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
                	Point leftPupil = detectPupil(faceMat.submat(eyesArray[0]));
                	Point rightPupil = detectPupil(faceMat.submat(eyesArray[1]));
                	
                	/*
                	Point tl = new Point(eyesArray[0].x, eyesArray[0].y + 0.33 * eyesArray[0].height);
                	Point br = new Point(eyesArray[0].br().x, eyesArray[0].br().y - 0.33 * eyesArray[0].height);
                	Rect leftEye = new Rect(tl, br);
                	
                	Point tl2 = new Point(eyesArray[1].x, eyesArray[1].y + 0.33 * eyesArray[1].height);
                	Point br2 = new Point(eyesArray[1].br().x, eyesArray[1].br().y - 0.33 * eyesArray[1].height);
                	Rect rightRoi = new Rect(tl2, br2);
                	*/
                	
                	if (leftPupil != null && rightPupil != null) {
                		Point[] leftCorners = detectCorners3(faceMat.submat(eyesArray[0]), true);
                		Point[] rightCorners = detectCorners3(faceMat.submat(eyesArray[1]), false);
                		
                		if (leftCorners != null && rightCorners != null) {
                        		//TODO: use a stable point instead of the center of boxes...
                        		double leftX = (eyesArray[0].tl().x + eyesArray[0].br().x)/2;
                            	double rightX = (eyesArray[1].tl().x + eyesArray[1].br().x)/2;
                            	double noseX = (nose.tl().x + nose.br().x)/2;

                            	double d = leftX - rightX;
                            	direction = "Face: " + (int)(((leftX - noseX) * 150 / d) - 75) + "°";
                            	
                            	double eyeLength = leftCorners[1].x - leftCorners[0].x;
                            	double dprime = leftCorners[1].x - (leftPupil.x + faceMat.submat(eyesArray[0]).cols()/10);

                            	directionEye = "Eyes : " + (int)((360 / eyeLength) * dprime - 180) + "°";                        	
                            	totalDir = "Total: " + ((int)((leftX - noseX) * 150 / d) - 75 + (int)((360 / eyeLength) * dprime - 180)) + "°";
                            	
                            	OscMessage myOscMessage = new OscMessage("/gaze");
                            	myOscMessage.add(id);
                            	myOscMessage.add(System.currentTimeMillis());
                            	myOscMessage.add(((int)((leftX - noseX) * 150 / d) - 75 + (int)((360 / eyeLength) * dprime - 180)));
                            	
                            	OscP5.flush(myOscMessage, myBroadcastLocation);
                            	
                            	//Draw the left pupil
                            	Core.circle(mGray, MyUtils.offset(MyUtils.offset(leftPupil, eyesArray[0].tl()), face.tl()), 3, PUPIL_COLOR, -1, 8, 0);

                    	} else {
                        	directionText("Corners not found.");
                        }
                	} else {
                    	directionText("Pupil not found.");
                    }
                	
		            //Draw the rectangles around left and right eyes.
		            Core.rectangle(mGray, MyUtils.offset(eyesArray[0].tl(), face.tl()), MyUtils.offset(eyesArray[0].br(), face.tl()), EYES_RECT_COLOR, 3);
		            Core.rectangle(mGray, MyUtils.offset(eyesArray[1].tl(), face.tl()), MyUtils.offset(eyesArray[1].br(), face.tl()), EYES_RECT_COLOR, 3);
                } else {
                	directionText("Eye(s) not found.");
                }
                
                //Draw the rectangle around the nose.
                Core.rectangle(mGray, MyUtils.offset(nose.tl(), face.tl()), MyUtils.offset(nose.br(), face.tl()), EYES_RECT_COLOR, 3);
            } else {
            	directionText("Nose not found.");
            }
            //Draw the rectangle around the face.
            Core.rectangle(mGray, face.tl(), face.br(), FACE_RECT_COLOR, 3);
        } else {
        	directionText("Face not found.");
        }
        	
    	ts = System.currentTimeMillis();
        Bitmap bmp = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mGray, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        te = System.currentTimeMillis();
        Log.d(TAG, "Creating Bitmap : " + (te - ts) + " ms.");
        long processingEnd = System.currentTimeMillis();
        Log.d(TAG, "Processing frame : " + (processingEnd - processingStart) + " ms.");
        return bmp;
    }

	private Rect detectFaces(Mat img) {
        MatOfRect faces = new MatOfRect();

        faceClassifier.detectMultiScale(img, faces, 1.5, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows()/5, img.cols()/5), new Size(img.rows(), img.cols()));

        if (faces.empty()) {
        	return null;
        } else {
        	return MyUtils.getBiggestRect(faces.toArray());
        }
	}

	public Rect detectNose(Mat face) {
		Rect roi = new Rect(new Point(face.cols() / 5, face.rows() / 5), new Point(4 * face.cols() / 5, 4 * face.rows() / 5));
		face = face.submat(roi);

		MatOfRect noses = new MatOfRect();
		
		noseClassifier.detectMultiScale(face, noses, 1.4, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(face.rows() / 3, face.rows() / 3), new Size(face.cols(), face.rows()));

		if (noses.empty()) {
			return null;
		} else {
        	return MyUtils.offset(MyUtils.getBiggestRect(noses.toArray()), roi.tl());
        }
	}
	
	private Rect[] detectEyes(Mat face) {
		Rect roi = new Rect(new Point(0, face.rows() / 6), new Point(face.cols(), face.rows() / 1.8));
        face = face.submat(roi);

        Mat rightArea = face.colRange(0, face.cols() / 2);
        Mat leftarea = face.colRange(face.cols() / 2, face.cols());

		MatOfRect rightEye = new MatOfRect();
		MatOfRect leftEye = new MatOfRect();

		eyeClassifier.detectMultiScale(rightArea, rightEye, 1.2, 1, 0, new Size(rightArea.cols() / 3, rightArea.cols() / 3), new Size(rightArea.rows(), rightArea.rows()));
		eyeClassifier.detectMultiScale(leftarea, leftEye, 1.2, 1, 0, new Size(leftarea.cols() / 3, leftarea.cols() / 3), new Size(leftarea.rows(), leftarea.rows()));

		if (rightEye.empty() || leftEye.empty()) {
			return null;
		}

		Rect[] eyesArray = new Rect[2];
		eyesArray[0] = MyUtils.offset(MyUtils.getBiggestRect(leftEye.toArray()), new Point(face.cols() / 2, roi.tl().y));
		eyesArray[1] = MyUtils.offset(MyUtils.getBiggestRect(rightEye.toArray()), roi.tl());
		
        return eyesArray;
	}
	
	private int findIrisRadius(Mat eye, Point pupil) {
		int d = 5;
		while ((int)pupil.x - d >= 0 && (int)pupil.x + d < eye.cols() && (eye.get((int)pupil.y + 10, (int)pupil.x - d)[0] < 100 || eye.get((int)pupil.y + 10, (int)pupil.x + d)[0] < 100)) {
			Log.d(TAG, "pupil center : " + pupil.x);
			Log.d(TAG, "d : " + d);
			Log.d(TAG, "eye.cols" + eye.cols());
			d += 5;
		}
		
		if ((int)pupil.x - d < 0 || (int)pupil.x + d >= eye.cols()) {
			return 0;
		} else {
			Core.circle(eye, pupil, d, PUPIL_COLOR, 1, 8, 0);
			return d;
		}
	}
		
	public static Point detectPupil(Mat eye) {
		Mat eyeClone = eye.clone();
		
		Rect roi = new Rect(new Point(0, eyeClone.rows() / 3), new Point(eyeClone.cols(), 2 * eyeClone.rows() / 3));
		eyeClone = eyeClone.submat(roi);
		
        Imgproc.blur(eyeClone, eyeClone, new Size(15, 15));
        
        double minBrightness = 255;        
        for (int a = 0; a < eyeClone.rows(); a = a + 5) {
        	for (int b = 0; b < eyeClone.cols(); b = b + 5) {
        		double[] pixel = eyeClone.get(a, b);

        		if (pixel[0] < minBrightness) {
        			minBrightness = pixel[0];
        		}
        	}
        }

        Imgproc.threshold(eyeClone, eyeClone, minBrightness + 15, 255, Imgproc.THRESH_BINARY_INV);
        
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(eyeClone, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
        
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

            MyUtils.offset(bar, roi.tl());
        }

        return bar;        
	}


	private Point[] detectCorners(Mat eye, boolean left, Point pupil) {
		Mat eyeClone = eye.clone();

		int d = findIrisRadius(eye, pupil);
		//eyeClone = eyeClone.rowRange(eyeClone.rows() / 4, 3 * eyeClone.rows() / 4);

		Log.d(TAG,"l : " + ((int)pupil.x - d));
		Log.d(TAG,"r : " + ((int)pupil.x + d));
		Mat leftArea = eyeClone.colRange(0, (int)pupil.x - d);
		Mat rightArea = eyeClone.colRange((int)pupil.x + d, eyeClone.cols());
		
		int minLeftBrightness = findMinBrightness(leftArea, 5);
		int minRightBrightness = findMinBrightness(rightArea, 5);
		
        Imgproc.threshold(leftArea, leftArea, minLeftBrightness + 15, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(rightArea, rightArea, minRightBrightness + 15, 255, Imgproc.THRESH_BINARY);
/*
        if (debugImage) {
        	MyUtils.saveImage(eye, "eye");
        	MyUtils.saveImage(leftArea, "leftArea");
        	MyUtils.saveImage(rightArea, "rightArea");
        	debugImage = false;
        }
*/
        Point[] corners = new Point[2];
        boolean found = false;
        for (int i = leftArea.rows() - 1; i >= 0  && !found; i--) {
			for (int j = 0; j < leftArea.cols() && !found; j++) {
				if (leftArea.get(i, j)[0] == 0) {
					if (left) {
						corners[1] = new Point(j, i);
				    	Core.circle(eye.colRange(0, (int)pupil.x - d), corners[1], 3, PUPIL_COLOR, -1, 8, 0);
				    	found = true;
					} else {
						corners[0] = new Point(j, i);
				    	Core.circle(eye.colRange(0, (int)pupil.x - d), corners[0], 3, PUPIL_COLOR, -1, 8, 0);
				    	found = true;
					}
				}
			}
		}
        if (!found) {
        	return null;
        }
        
        found = false;
        for (int i = rightArea.rows() - 1; i >= 0  && !found; i--) {
			for (int j = rightArea.cols() - 1; j >= 0 && !found; j--) {
				if (rightArea.get(i, j)[0] == 0) {
					if (left) {
						corners[0] = MyUtils.offset(new Point(j, i), new Point((int)pupil.x + d + 5, 0));
				    	Core.circle(eye.colRange((int)pupil.x + d, eyeClone.cols()), new Point(j, i), 3, PUPIL_COLOR, -1, 8, 0);
				    	found = true;
					} else {
						corners[1] = MyUtils.offset(new Point(j, i), new Point((int)pupil.x + d + 5, 0));
				    	Core.circle(eye.colRange((int)pupil.x + d, eyeClone.cols()), new Point(j, i), 3, PUPIL_COLOR, -1, 8, 0);
				    	found = true;
					}
				}
			}
		}
        if (!found) {
        	return null;
        }

        return corners;        
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
	
	public void draw(Canvas canvas, float offsetx, float offsety) {
        canvas.drawText(direction, 10 + offsetx, 10 + 50 + offsety, PAINT);
        canvas.drawText(directionEye, 10 + offsetx, 10 + 50 + 50 + offsety, PAINT);
        canvas.drawText(totalDir, 10 + offsetx, 10 + 50 + 50 + 50 + offsety, PAINT);
    }
	
	private void directionText(String message) {
		direction = message;
    	directionEye = message;
    	totalDir = message;
    	Log.d(TAG, message);
	}
	
	private int findMinBrightness(Mat img, int accuracy) {
		double brightness = 255;
		
		for (int i = 0; i < img.rows(); i+=accuracy) {
			for (int j = 0; j < img.cols(); j+=accuracy) {
				if (img.get(i, j)[0] < brightness) {
					brightness = img.get(i, j)[0];
				}
			}
		}
		
		return (int)brightness;
	}
	
	private Point[] detectCorners3(Mat eye, boolean left) {
		Mat eyeClone = eye.clone();

		Mat leftCornerArea = eyeClone
				.colRange(0, (int) (eyeClone.cols() / 3.5));
		Mat rightCornerArea = eyeClone.colRange(
				(int) (2.5 * eyeClone.cols() / 3.5), eyeClone.cols());

		Imgproc.blur(leftCornerArea, leftCornerArea, new Size(3, 3));
		Imgproc.blur(rightCornerArea, rightCornerArea, new Size(3, 3));

		double minBrightness = 255;

		for (int a = 0; a < leftCornerArea.rows(); a = a + 5) {
			for (int b = 0; b < leftCornerArea.cols(); b = b + 5) {
				double[] pixel = leftCornerArea.get(a, b);

				if (pixel[0] < minBrightness) {
					minBrightness = pixel[0];
				}
			}
		}

		Imgproc.threshold(leftCornerArea, leftCornerArea, minBrightness, 255,
				Imgproc.THRESH_BINARY_INV);

		Point leftCorner = null;
		boolean found = false;
		for (int a = 0; a < leftCornerArea.cols() && !found; a++) {
			for (int b = 0; b < leftCornerArea.rows() && !found; b++) {
				double[] pixel = leftCornerArea.get(b, a);

				if (pixel[0] == 255) {
					found = true;
					leftCorner = new Point(a, b);
				}
			}
		}

		minBrightness = 255;

		for (int a = 0; a < rightCornerArea.rows(); a = a + 5) {
			for (int b = 0; b < rightCornerArea.cols(); b = b + 5) {
				double[] pixel = rightCornerArea.get(a, b);

				if (pixel[0] < minBrightness) {
					minBrightness = pixel[0];
				}
			}
		}

		Imgproc.threshold(rightCornerArea, rightCornerArea, minBrightness, 255,
				Imgproc.THRESH_BINARY_INV);

		Point rightCorner = null;
		found = false;
		for (int a = rightCornerArea.cols() - 1; a >= 0 && !found; a--) {
			for (int b = 0; b < rightCornerArea.rows() && !found; b++) {
				double[] pixel = rightCornerArea.get(b, a);
				if (pixel[0] == 255) {
					found = true;
					rightCorner = new Point(a + 3 * eyeClone.cols() / 4, b);
				}
			}
		}

		if (leftCorner == null || rightCorner == null) {
			return null;
		}

		Point[] cornersArray = new Point[2];

		Core.circle(eye, leftCorner, 3, PUPIL_COLOR, -1, 8, 0);
		Core.circle(eye, rightCorner, 3, PUPIL_COLOR, -1, 8, 0);

		if (left) {
			cornersArray[0] = leftCorner;
			cornersArray[1] = rightCorner;
		} else {
			cornersArray[0] = rightCorner;
			cornersArray[1] = leftCorner;
		}

		return cornersArray;
	}
}
