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

	private static final Paint PAINT = new Paint();
	private static final Scalar FACE_RECT_COLOR = new Scalar(0);
	private static final Scalar EYES_RECT_COLOR = new Scalar(51);
	private static final Scalar PUPIL_COLOR = new Scalar(255);

	private Mat mGray;

	private int xeyeDirection;
	private int yeyeDirection;

	private CascadeClassifier faceClassifier;
	private CascadeClassifier eyeClassifier;
	private CascadeClassifier noseClassifier;
	
	///// Calibration data
	boolean toTLCalibrate;
	boolean isTLCalibrated;
	int TLxDirection;
	int TLyDirection;
	
	boolean toBRCalibrate;
	boolean isBRCalibrated;
	int BRxDirection;
	int BRyDirection;
	
	private String msg = null;

	public Tracker(Context context) {
		faceClassifier = new CascadeClassifier();
		loadClassifier(context, faceClassifier, R.raw.haarcascade_frontalface_alt2, "haarcascade_frontalface_alt2.xml");
		eyeClassifier = new CascadeClassifier();
		loadClassifier(context, eyeClassifier, R.raw.haarcascade_eye, "haarcascade_eye");
		noseClassifier = new CascadeClassifier();
		loadClassifier(context, noseClassifier, R.raw.haarcascade_mcs_nose, "haarcascade_mcs_nose.xml");

		xeyeDirection = 0;
		yeyeDirection = 0;

		mGray = new Mat();

		PAINT.setColor(Color.YELLOW);
		PAINT.setTextSize(50);

		myBroadcastLocation = new NetAddress("192.168.32.130", 32000);
		
		toTLCalibrate = false;
		isTLCalibrated = false;
		TLxDirection = 0;
		TLyDirection = 0;
		
		toBRCalibrate = false;
		isBRCalibrated = false;
		BRxDirection = 0;
		BRyDirection = 0;
	}

	protected Bitmap processFrame(VideoCapture capture) {
		try {
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

						Point tl = new Point(eyesArray[0].x - 0.1 * eyesArray[0].width, eyesArray[0].y + 0.33 * eyesArray[0].height);
						Point br = new Point(eyesArray[0].br().x + 0.1 * eyesArray[0].width, eyesArray[0].br().y - 0.33 * eyesArray[0].height);
						Rect leftEye = new Rect(tl, br);

						Point tl2 = new Point(eyesArray[1].x - 0.1 * eyesArray[1].width, eyesArray[1].y + 0.33 * eyesArray[1].height);
						Point br2 = new Point(eyesArray[1].br().x + 0.1 * eyesArray[1].width, eyesArray[1].br().y - 0.33 * eyesArray[1].height);
						Rect rightRoi = new Rect(tl2, br2);

						if (leftPupil != null && rightPupil != null) {
							Point[] leftCorners = detectCorners3(faceMat.submat(leftEye), true);
							Point[] rightCorners = detectCorners3(faceMat.submat(rightRoi), false);

							if (leftCorners != null && rightCorners != null) {
								if (!isTLCalibrated || !isBRCalibrated) {
									if (!isTLCalibrated) {
										msg = "Not TL-calibrated";
									} else {
										msg = "Not BR-calibrated";
									}
								}
								
								double leftX = (eyesArray[0].tl().x + eyesArray[0].br().x) / 2;
								double rightX = (eyesArray[1].tl().x + eyesArray[1].br().x) / 2;
								double noseX = (nose.tl().x + nose.br().x) / 2;

								double d = leftX - rightX;
								int faceDirection = (int) (((leftX - noseX) * 150 / d) - 75);

								//
								double leftEyeLength = leftCorners[1].x - leftCorners[0].x;
								double leftRatio = (leftPupil.x - leftCorners[0].x)/leftEyeLength;
								
								//double rightEyeLength = rightCorners[1].x - rightCorners[0].x;
								//double rightRatio = (rightPupil.x - rightCorners[0].x)/rightEyeLength;

								int leftXeyeDirection = (int) ((-45/0.4) * leftRatio + (0.5 * 45 / 0.4));
								//int rightXeyeDirection = (int) ((-45/0.4) * rightRatio + (0.5 * 45 / 0.4));
								
								//Log.d(TAG, "left and right ratio" + leftRatio + ", " + rightRatio);
								//Log.d(TAG, "left and right direction" + leftXeyeDirection + ", " + rightXeyeDirection);
								
								xeyeDirection = leftXeyeDirection;
								/*
								if (Math.abs(leftXeyeDirection) > 60 && Math.abs(rightXeyeDirection) > 60) {
									xeyeDirection = (leftXeyeDirection + rightXeyeDirection)/2;
								} else if (Math.abs(leftXeyeDirection) > 60) {
									xeyeDirection = rightXeyeDirection;
								} else if (Math.abs(rightXeyeDirection) > 60) {
									xeyeDirection = leftXeyeDirection;
								} else {
									xeyeDirection = (leftXeyeDirection + rightXeyeDirection)/2;
								}
								*/
								xeyeDirection = xeyeDirection + faceDirection;
															
								if (toTLCalibrate) {
									TLxDirection = xeyeDirection;
									Log.d(TAG, "TLx is set to :" + xeyeDirection);
									toTLCalibrate = false;
									isTLCalibrated = true;
								}
								if (toBRCalibrate) {
									BRxDirection = xeyeDirection;
									Log.d(TAG, "BRx is set to :" + xeyeDirection);
									toBRCalibrate = false;
									isBRCalibrated = true;
								}
								
								if (isTLCalibrated && isBRCalibrated) {
									double xPosition = (double)(xeyeDirection-TLxDirection) / (BRxDirection - TLxDirection);
									double yPosition = 0;
									if (xPosition >= 0 && xPosition <= 1) {
										OscMessage myOscMessage = new OscMessage("/gaze");
										myOscMessage.add(id);
										myOscMessage.add(xPosition);
										myOscMessage.add(yPosition);

										msg = "Ok.";
										OscP5.flush(myOscMessage, myBroadcastLocation);
									} else {
										msg = "Bad value.";
									}
								}
								
								
								
								/*
								// TODO: use a stable point instead of the center of
								// boxes...
								double leftX = (eyesArray[0].tl().x + eyesArray[0].br().x) / 2;
								double rightX = (eyesArray[1].tl().x + eyesArray[1].br().x) / 2;
								double noseX = (nose.tl().x + nose.br().x) / 2;

								double d = leftX - rightX;
								faceDirection = (int) (((leftX - noseX) * 150 / d) - 75);

								//
								double leftEyeLength = leftCorners[1].x - leftCorners[0].x;
								double leftRatio = (leftPupil.x - leftCorners[0].x)/leftEyeLength;
								
								double rightEyeLength = rightCorners[1].x - rightCorners[0].x;
								double rightRatio = (rightPupil.x - rightCorners[0].x)/rightEyeLength;

								int leftXeyeDirection = (int) ((-45/0.4) * leftRatio + (0.5 * 45 / 0.4));
								int rightXeyeDirection = (int) ((-45/0.4) * rightRatio + (0.5 * 45 / 0.4));
								
								Log.d(TAG, "left and right ratio" + leftRatio + ", " + rightRatio);
								Log.d(TAG, "left and right direction" + leftXeyeDirection + ", " + rightXeyeDirection);
								
								if (Math.abs(leftXeyeDirection) > 60 && Math.abs(rightXeyeDirection) > 60) {
									xeyeDirection = (leftXeyeDirection + rightXeyeDirection)/2;
								} else if (Math.abs(leftXeyeDirection) > 60) {
									xeyeDirection = rightXeyeDirection;
								} else if (Math.abs(rightXeyeDirection) > 60) {
									xeyeDirection = leftXeyeDirection;
								} else {
									xeyeDirection = (leftXeyeDirection + rightXeyeDirection)/2;
								}
								
								totalDirection = xeyeDirection + faceDirection;

								OscMessage myOscMessage = new OscMessage("/gaze");
								myOscMessage.add(id);
								myOscMessage.add(System.currentTimeMillis());
								myOscMessage.add(seatNumber);
								myOscMessage.add(totalDirection);

								OscP5.flush(myOscMessage, myBroadcastLocation);
								*/
								// Draw the left pupil
								Core.circle(mGray, MyUtils.offset(MyUtils.offset(leftPupil, eyesArray[0].tl()), face.tl()), 3, PUPIL_COLOR, -1, 8, 0);
								Core.circle(mGray, MyUtils.offset(MyUtils.offset(rightPupil, eyesArray[1].tl()), face.tl()), 3, PUPIL_COLOR, -1, 8, 0);

							} else {
								msg = "Eye Corners not found.";
							}
						} else {
							msg = "Pupil not found.";
						}

						// Draw the rectangles around left and right eyes.
						Core.rectangle(mGray, MyUtils.offset(eyesArray[0].tl(), face.tl()), MyUtils.offset(eyesArray[0].br(), face.tl()), EYES_RECT_COLOR, 3);
						Core.rectangle(mGray, MyUtils.offset(eyesArray[1].tl(), face.tl()), MyUtils.offset(eyesArray[1].br(), face.tl()), EYES_RECT_COLOR, 3);
					} else {
						msg = "Eye(s) not found.";
					}

					// Draw the rectangle around the nose.
					Core.rectangle(mGray, MyUtils.offset(nose.tl(), face.tl()), MyUtils.offset(nose.br(), face.tl()), EYES_RECT_COLOR, 3);
				} else {
					msg = "Nose not found.";
				}
				// Draw the rectangle around the face.
				Core.rectangle(mGray, face.tl(), face.br(), FACE_RECT_COLOR, 3);
			} else {
				msg = "Face not found.";
			}

			ts = System.currentTimeMillis();
			Bitmap bmp = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);

			try {
				Utils.matToBitmap(mGray, bmp);
			} catch (Exception e) {
				Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
				bmp.recycle();
				bmp = null;
			}
			te = System.currentTimeMillis();
			Log.d(TAG, "Creating Bitmap : " + (te - ts) + " ms.");
			long processingEnd = System.currentTimeMillis();
			Log.d(TAG, "Processing frame : " + (processingEnd - processingStart) + " ms.");
			return bmp;
		} catch (Exception e) {
			return null;
		}
	}

	private Rect detectFaces(Mat img) {
		MatOfRect faces = new MatOfRect();

		faceClassifier.detectMultiScale(img, faces, 1.3, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows() / 5, img.cols() / 5),
				new Size(img.rows(), img.cols()));

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

		noseClassifier.detectMultiScale(face, noses, 1.3, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(face.rows() / 3, face.rows() / 3), new Size(face.rows(),
				face.cols()));

		if (noses.empty()) {
			return null;
		} else {
			return MyUtils.offset(MyUtils.getBiggestRect(noses.toArray()), roi.tl());
		}
	}

	private Rect[] detectEyes(Mat face) {
		Rect leftRoi = new Rect(new Point(0, face.rows() / 6), new Point(face.cols() / 2, face.rows() / 1.5));
		Rect rightRoi = new Rect(new Point(face.cols() / 2, face.rows() / 6), new Point(face.cols(), face.rows() / 1.5));

		Mat rightArea = face.submat(rightRoi);
		Mat leftarea = face.submat(leftRoi);

		MatOfRect rightEye = new MatOfRect();
		MatOfRect leftEye = new MatOfRect();

		eyeClassifier.detectMultiScale(rightArea, rightEye, 1.2, 1, 0, new Size(rightArea.cols() / 3, rightArea.cols() / 3), new Size(rightArea.rows(),
				rightArea.rows()));
		eyeClassifier.detectMultiScale(leftarea, leftEye, 1.2, 1, 0, new Size(leftarea.cols() / 3, leftarea.cols() / 3),
				new Size(leftarea.rows(), leftarea.rows()));

		if (rightEye.empty() || leftEye.empty()) {
			return null;
		}

		Rect[] eyesArray = new Rect[2];
		eyesArray[0] = MyUtils.offset(MyUtils.getBiggestRect(leftEye.toArray()), leftRoi.tl());
		eyesArray[1] = MyUtils.offset(MyUtils.getBiggestRect(rightEye.toArray()), rightRoi.tl());

		return eyesArray;
	}

	public static Point detectPupil(Mat eye) {
		Mat eyeClone = eye.clone();

		Rect roi = new Rect(new Point(0, eyeClone.rows() / 4), new Point(eyeClone.cols(), 3 * eyeClone.rows() / 4));
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

		Imgproc.threshold(eyeClone, eyeClone, minBrightness + 10, 255, Imgproc.THRESH_BINARY_INV);

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

			for (Point element : pupilContour) {
				bar.x += element.x;
				bar.y += element.y;
			}

			bar.x /= pupilContour.length;
			bar.y /= pupilContour.length;

			MyUtils.offset(bar, roi.tl());
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

	public void draw(Canvas canvas, float offsetx, float offsety) {
		if (msg != null) {
			canvas.drawText(msg, 10 + offsetx, 10 + 50 + offsety, PAINT);
			msg = null;
		}
	}

	private Point[] detectCorners3(Mat eye, boolean left) {
		Mat eyeClone = eye.clone();

		Mat leftCornerArea = eyeClone.colRange(0, (int) (eyeClone.cols() / 3.5));
		Mat rightCornerArea = eyeClone.colRange((int) (2.5 * eyeClone.cols() / 3.5), eyeClone.cols());

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

		Imgproc.threshold(leftCornerArea, leftCornerArea, minBrightness, 255, Imgproc.THRESH_BINARY_INV);

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

		Imgproc.threshold(rightCornerArea, rightCornerArea, minBrightness, 255, Imgproc.THRESH_BINARY_INV);

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

		cornersArray[0] = leftCorner;
		cornersArray[1] = rightCorner;
		
		return cornersArray;
	}
	
	public void calibrateTL() {
		toTLCalibrate = true;
	}
	
	public void calibrateBR() {
		toBRCalibrate = true;		
	}
}
