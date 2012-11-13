import netP5.NetInfo;
import oscP5.OscEventListener;
import oscP5.OscMessage;
import oscP5.OscP5;
import oscP5.OscStatus;

public class Server {
	OscP5 oscP5;
	
	OscEventListener l = new OscEventListener() {		
		@Override
		public void oscStatus(OscStatus arg0) {
			// TODO Auto-generated method stub
			
		}
		
		@Override
		public void oscEvent(OscMessage arg0) {
			System.out.println("### received an osc message.");
			System.out.println("\taddrpattern: " + arg0.addrPattern());
			System.out.println("\ttypetag: " + arg0.typetag());
			
			switch (arg0.typetag()) {
			case "i":
				int angle = arg0.get(0).intValue();
				System.out.println("Someone is looking at " + angle + " degree.\n");
				break;
			default:
				break;
			}
		}
	};
	
	public Server() {
		oscP5 = new OscP5(this, 32000);
		NetInfo.wan();
		oscP5.addListener(l);
	}
}
