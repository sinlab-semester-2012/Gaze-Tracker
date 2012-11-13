import netP5.NetAddress;
import oscP5.OscMessage;
import oscP5.OscP5;


public class Client {
	private NetAddress myBroadcastLocation;
	
	public static void main(String[] args) {
		new Client();
	}

	public Client() {
        myBroadcastLocation = new NetAddress("192.168.56.1",32000);
        
        OscMessage myOscMessage = new OscMessage("/gaze");
    	myOscMessage.add(12);
    	OscP5.flush(myOscMessage, myBroadcastLocation);
    	
	}
}
