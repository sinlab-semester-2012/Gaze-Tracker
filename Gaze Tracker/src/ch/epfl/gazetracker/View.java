package ch.epfl.gazetracker;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

class View extends SurfaceView implements SurfaceHolder.Callback, Runnable {
	private static final String TAG = "View";

	private VideoCapture mCamera;
	private FpsMeter mFps;
	private SurfaceHolder mHolder;
	private Tracker mTracker;

	public View(Context context) {
		super(context);

		Log.i(TAG, "Instantiated new " + this.getClass());

		mFps = new FpsMeter();
		mHolder = getHolder();
		mHolder.addCallback(this);
		mTracker = new Tracker(context);
	}

	public boolean openCamera() {
		Log.i(TAG, "openCamera");

		synchronized (this) {
			releaseCamera();
			mCamera = new VideoCapture(Highgui.CV_CAP_ANDROID + 1);

			if (!mCamera.isOpened()) {
				mCamera.release();
				mCamera = null;
				Log.e(TAG, "Failed to open native camera");
				return false;
			}
		}

		return true;
	}

	public void releaseCamera() {
		Log.i(TAG, "releaseCamera");

		synchronized (this) {
			if (mCamera != null) {
				mCamera.release();
				mCamera = null;
			}
		}
	}

	public void setupCamera(int width, int height) {
		Log.i(TAG, "setupCamera(" + width + ", " + height + ")");
		synchronized (this) {
			if (mCamera != null && mCamera.isOpened()) {
				List<Size> sizes = mCamera.getSupportedPreviewSizes();
				int mFrameWidth = width;
				int mFrameHeight = height;

				// selecting optimal camera preview size
				double minDiff = Double.MAX_VALUE;
				for (Size size : sizes) {
					if (Math.abs(size.height - height) < minDiff) {
						mFrameWidth = (int) size.width;
						mFrameHeight = (int) size.height;
						minDiff = Math.abs(size.height - height);
					}
				}

				mCamera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, mFrameWidth);
				mCamera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, mFrameHeight);
			}
		}

	}

	@Override
	public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
		Log.i(TAG, "surfaceChanged");
		setupCamera(width, height);
	}

	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		Log.i(TAG, "surfaceCreated");
		(new Thread(this)).start();
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder holder) {
		Log.i(TAG, "surfaceDestroyed");
		releaseCamera();
	}

	@Override
	public void run() {
		Log.i(TAG, "Starting processing thread");
		mFps.init();

		while (true) {
			Bitmap bmp = null;

			synchronized (this) {
				if (mCamera == null) {
					break;
				}

				if (!mCamera.grab()) {
					Log.e(TAG, "mCamera.grab() failed");
					break;
				}

				bmp = mTracker.processFrame(mCamera);

				mFps.measure();
			}

			if (bmp != null) {
				Canvas canvas = mHolder.lockCanvas();
				if (canvas != null) {
					canvas.drawBitmap(bmp, (canvas.getWidth() - bmp.getWidth()) / 2, (canvas.getHeight() - bmp.getHeight()) / 2, null);
					mFps.draw(canvas, (canvas.getWidth() - bmp.getWidth()) / 2, (canvas.getHeight() - bmp.getHeight()) / 2);
					mTracker.draw(canvas, (canvas.getWidth() - bmp.getWidth()) / 2, (canvas.getHeight() - bmp.getHeight()) / 2 + 50);
					mHolder.unlockCanvasAndPost(canvas);
				}
				bmp.recycle();
			}
		}

		Log.i(TAG, "Finishing processing thread");
	}
	
	public void calibrateTL() {
		mTracker.calibrateTL();
	}
	
	public void calibrateBR() {
		mTracker.calibrateBR();
	}
}