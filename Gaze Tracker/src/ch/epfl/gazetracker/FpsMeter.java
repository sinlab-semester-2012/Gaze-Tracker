package ch.epfl.gazetracker;

import java.text.DecimalFormat;

import org.opencv.core.Core;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

public class FpsMeter {
	private static final String TAG = "FpsMeter";
	private static final int STEP = 20;
	private static final double FREQ = Core.getTickFrequency();
	private static final DecimalFormat FORMAT = new DecimalFormat("0.00");
	private static final Paint PAINT = new Paint();

	private int framesCounter;
	private long prevFrameTime;
	private String strfps;

	public void init() {
		framesCounter = 0;
		prevFrameTime = Core.getTickCount();
		strfps = "";

		PAINT.setColor(Color.BLUE);
		PAINT.setTextSize(50);
	}

	public void measure() {
		framesCounter = (framesCounter + 1) % STEP;

		if (framesCounter == 0) {
			long time = Core.getTickCount();
			double fps = STEP * FREQ / (time - prevFrameTime);
			prevFrameTime = time;
			strfps = FORMAT.format(fps) + " FPS";
			Log.i(TAG, strfps);
		}
	}

	public void draw(Canvas canvas, float offsetx, float offsety) {
		canvas.drawText(strfps, 10 + offsetx, 10 + 50 + offsety, PAINT);
	}
}
