package ch.epfl.gazetracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

public class MyUtils {
	public static void saveImage(Mat img, String filename) {
		Bitmap bmp = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);

		try {
			Utils.matToBitmap(img, bmp);

			String state = Environment.getExternalStorageState();

			if (Environment.MEDIA_MOUNTED.equals(state)) {
				// We can read and write the media

				File baseDir = new File(Environment.getExternalStorageDirectory() + "/Android/data/ch.epfl.gazetracker");
				baseDir.mkdir();

				File myBmp = new File(baseDir, filename + ".png");
				OutputStream os = new FileOutputStream(myBmp);

				Log.e("saver", "saving here : " + myBmp.getAbsolutePath());

				bmp.compress(Bitmap.CompressFormat.PNG, 100, os);
				os.flush();
				os.close();

			} else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
				Log.e("saver", "can read but not write");
				// We can only read the media

			} else {
				Log.e("saver", "can't do anything");
				// Something else is wrong. It may be one of many other states,
				// but all we need
				// to know is we can neither read nor write
			}
		} catch (Exception e) {
			Log.e("saver", "an exception occur : " + e.getMessage());
			e.getStackTrace();
			bmp.recycle();
			bmp = null;
		}
	}

	public static Rect getBiggestRect(Rect[] rArray) {
		Rect rect = rArray[0];
		for (int i = 1; i < rArray.length; i++) {
			if (rArray[i].area() > rect.area()) {
				rect = rArray[i];
			}
		}
		return rect;
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
