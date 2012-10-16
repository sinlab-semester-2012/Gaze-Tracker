package ch.epfl.gazetracker;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class Tracker {
	static boolean pic = true;
	
	public static Rect[] detectFaces(Mat img, CascadeClassifier facesClassifier) {
        MatOfRect faces = new MatOfRect();
        
        facesClassifier.detectMultiScale(img, faces, 1.2, 1, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows()/3, img.cols()/3), new Size(img.rows(), img.cols()));
        
        return faces.toArray();
	}
	
	public static Rect[] detectEyes(Mat face, CascadeClassifier eyesClassifier) {
		// Manual approximation of the eyes area
		Rect roi = new Rect(new Point(0, face.rows() / 5), new Point(face.cols(), face.rows() / 1.8));
        face = face.submat(roi);
        
        Mat rightArea = face.colRange(0, face.cols() / 2);
        Mat leftarea = face.colRange(face.cols() / 2, face.cols());

		MatOfRect rightEye = new MatOfRect();
		MatOfRect leftEye = new MatOfRect();

		eyesClassifier.detectMultiScale(rightArea, rightEye, 1.2, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(rightArea.cols() / 3, rightArea.cols() / 3), new Size(rightArea.rows(), rightArea.rows()));
		eyesClassifier.detectMultiScale(leftarea, leftEye, 1.2, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(leftEye.cols() / 3, leftEye.cols() / 3), new Size(leftEye.rows(), leftEye.rows()));

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
	
	public static Rect detectNose(Mat face, CascadeClassifier noseClassifier) {
		Rect roi = new Rect(new Point(0, face.rows() / 4), new Point(face.cols(), 3 * face.rows() / 4));
		face = face.submat(roi);
		
		MatOfRect noses = new MatOfRect();
		
		noseClassifier.detectMultiScale(face, noses, 1.2, 1, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(0, 0), new Size(face.cols(), face.rows()));

		if (noses.toArray().length == 0) {
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
        if (pic) {
        	MyUtils.saveImage(eyeClone, "not threshold eye");
        }
        
        Imgproc.threshold(eyeClone, eyeClone, minBrightness + 2, 255, Imgproc.THRESH_BINARY);
        
        if (pic) {
        	MyUtils.saveImage(eyeClone, "threshold eye");
        	pic = false;
        }
        
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
}
