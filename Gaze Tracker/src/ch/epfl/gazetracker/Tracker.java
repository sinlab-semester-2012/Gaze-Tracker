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
        
        facesClassifier.detectMultiScale(img, faces, 1.2, 5, Objdetect.CASCADE_SCALE_IMAGE, new Size(img.rows()/3, img.cols()/3), new Size(img.rows(), img.cols()));
        
        return faces.toArray();
	}
	
	public static Rect detectOneEye(Mat face, CascadeClassifier eyesClassifier) {
		MatOfRect eyes = new MatOfRect();
        
        eyesClassifier.detectMultiScale(face, eyes, 1.2, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(face.rows()/2, face.cols()/2), new Size(face.rows(), face.cols()));
        
        return eyes.toArray()[0];
	}
	
	public static Rect[] detectEyes(Mat face, CascadeClassifier eyesClassifier) {
		MatOfRect eyes = new MatOfRect();
        
		eyesClassifier.detectMultiScale(face, eyes, 1.2, 3, 0, new Size(face.cols() / 6, face.cols() / 6), new Size(face.rows(), face.rows()));
        
        return eyes.toArray();
	}
	
	public static Point detectPupil(Mat eye) {
		
		Mat eyeClone = eye.clone();
		
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
        }

        return bar;
        
	}
}
