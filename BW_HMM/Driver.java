import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class Driver {

	//Just change these two constants for a submission
	private static final String PROBLEM_NUMBER = "2";
	private static final String SUBMISSION_NAME = "testprob2_11_02";

	// Team and project specific constants 
	private static final String TEAM_NUMBER = "81";
	private static final String TESTING_FILE = PROBLEM_NUMBER + ".spice.public.test";
	private static final String TRAINING_FILE = PROBLEM_NUMBER + ".spice.train";
	private static final String URL_BASE = "http://spice.lif.univ-mrs.fr/submit.php?user=" + TEAM_NUMBER + "&problem=" + PROBLEM_NUMBER + "&submission=" + SUBMISSION_NAME + "&";

	//HMM Tuning parameters
	private static final int BATCH_SIZE = 25;
	private static final int HIDDEN_STATES = 5; 
	private static final int MAX_ITERATION = 250;
	private static final int RETRY_PARAMETER = 50;
	private static final double TOLERANCE = Math.exp(-4); 
	private static final double LEARNING_RATE = 0.1; // unused for now
	private static int numSymbols = -1;

	public static void main(String args[]) throws IOException {

		// Get a trained hmm
		//HMM hmm = train(); 

		// Load a presaved HMM
		HMM hmmFile = loadHmmFromFile("TrainedHMMProblem2_11hs_15bs");
		hmmFile.print();
		// Predict using this hmm
		predictDriver(hmmFile);

	}

	private static HMM loadHmmFromFile(String filename) throws IOException {
		BufferedReader reader = null;
		try {
			File file = new File(filename);
			reader = new BufferedReader(new FileReader(file));
			reader.readLine(); // Skip first line
			String text = reader.readLine();
			int numStates = Integer.valueOf(text.split(" ")[0]);
			numSymbols = Integer.valueOf(text.split(" ")[1]);
			double tmpInitial[] = new double[numStates];
			double tmpTransition[][] = new double[numStates][numStates];
			double tmpEmission[][] = new double[numStates][numSymbols];

			for (int i = 0; i < numStates; i++) {
				text = reader.readLine();
				tmpInitial[i] = Double.valueOf(text);
			}
			for (int i = 0; i < numStates; i++) {
				text = reader.readLine();
				String[] vals = text.split(" ");
				for (int j = 0; j < numStates; j++) {
					tmpTransition[i][j] = Double.valueOf(vals[j]); 
				}
			}
			for (int i = 0; i < numStates; i++) {
				text = reader.readLine();
				String[] vals = text.split(" ");
				for (int j = 0; j < numSymbols; j++) {
					tmpEmission[i][j] = Double.valueOf(vals[j]); 
				}
			}
			HMM newHmm = new HMM(numStates, numSymbols, MAX_ITERATION, TOLERANCE);
			newHmm.setPi(tmpInitial);
			newHmm.setA(tmpTransition);
			newHmm.setB(tmpEmission);
			return newHmm;
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (reader!=null) {
				reader.close();
			}
		}
		return null;
	}

	private static HMM train() throws IOException {
		BufferedReader reader = null;
		File file = new File(TRAINING_FILE);
		try {
			reader = new BufferedReader(new FileReader(file));
			String text = reader.readLine();
			int numLines = Integer.valueOf(text.split(" ")[0]);
			numSymbols = Integer.valueOf(text.split(" ")[1]) + 1;
			HMM hmm = new HMM(HIDDEN_STATES, numSymbols, MAX_ITERATION, TOLERANCE);
			System.out.println("Begin training " + TRAINING_FILE + " having " + numLines + " lines and " + numSymbols + " symbols(including endline).");
			int batch = 1;
			int linecount = 0;
			while (linecount + BATCH_SIZE < numLines) {
				List<Integer> obs = new ArrayList<Integer>();
				for (int it = 0; it < BATCH_SIZE; it++) {
					String[] l = reader.readLine().split(" "); linecount++;
					int seqLen = Integer.parseInt(l[0]);
					for (int j = 0; j < seqLen ; j++) {
						obs.add(Integer.parseInt(l[j+1]));
					}
					obs.add(numSymbols-1);
				}
				System.out.println("Training batch " + batch);
				int[] obsArr = new int[obs.size()];
				for (int i=0; i < obsArr.length; i++)
					obsArr[i] = obs.get(i).intValue();

				double pisum = 0;
				int tries = 0;
				HMM temp_hmm = null;
				while (pisum == 0 && tries < RETRY_PARAMETER) {
					temp_hmm = new HMM(HIDDEN_STATES, numSymbols, MAX_ITERATION, TOLERANCE); 
					temp_hmm.setA(hmm.getA());
					temp_hmm.setB(hmm.getB());
					temp_hmm.setPi(hmm.getPi());
					temp_hmm.train(obsArr);
					for (double val : temp_hmm.getPi())
						pisum += val;
					tries++;
				} if (tries == RETRY_PARAMETER) throw new Exception("Sorry Brother");

				hmm.setA(mixWt(hmm.getA(),temp_hmm.getA(), 1/batch));
				hmm.setB(mixWt(hmm.getB(),temp_hmm.getB(), 1/batch));
				hmm.setPi(mixWt(hmm.getPi(),temp_hmm.getPi(), 1/batch));
				batch++;
			}
			System.out.println("Finished training " + linecount + " lines");
			hmm.print();
			hmm.store("TrainedHMMProblem" + PROBLEM_NUMBER + "_" + HIDDEN_STATES + "hs_" + BATCH_SIZE + "bs");
			return hmm;
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				reader.close();
			}
		}
		return null;
	}

	private static  void predictDriver(HMM hmm) throws IOException {
		File file = new File(TESTING_FILE);
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String firstFilePrefix = reader.readLine();
		String rankString = getRankingString(firstFilePrefix, hmm);
		int prefixNumber = 1;
		System.out.println("Prefix number: " + prefixNumber  + " Ranking: " + rankString + " Prefix: " + firstFilePrefix);
		if (reader != null) {
			reader.close();
		}
		URL obj = new URL(getURL(firstFilePrefix, prefixNumber, rankString));
		HttpURLConnection con = (HttpURLConnection) obj.openConnection();
		con.setRequestMethod("GET");
		BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
		String inputLine;
		StringBuffer response = new StringBuffer();
		while ((inputLine = in.readLine()) != null) {
			response.append(inputLine);
		}
		in.close();
		while (con.getResponseCode() == HttpURLConnection.HTTP_OK) {
			if(response.toString().split(" ").equals("[Error]")) {
				System.out.println(response.toString());
				break;
			}
			rankString = getRankingString(response.toString(), hmm);
			prefixNumber++;
			System.out.println("Prefix number: " + prefixNumber  + " Ranking: " + rankString + " Prefix: " + response.toString());
			if (reader != null) {
				reader.close();
			}
			obj = new URL(getURL(response.toString(), prefixNumber, rankString));
			con = (HttpURLConnection) obj.openConnection();
			con.setRequestMethod("GET");

			in = new BufferedReader(new InputStreamReader(con.getInputStream()));
			response = new StringBuffer();
			while ((inputLine = in.readLine()) != null) {
				response.append(inputLine);
			}
			in.close();
			System.out.println(response.toString());
			if (response.toString().split(" ")[0].equals("[Success]"))
				break;
		}

	}

	private static String getURL(String prefix, int prefixNumber, String rank) {
		return URL_BASE + "prefix=" + prefix.replace(" ", "%20") + "&prefix_number=" + prefixNumber + "&ranking=" + rank.replace(" ", "%20");
	}

	private static String getRankingString(String firstFilePrefix, HMM hmm) {
		String[] l = firstFilePrefix.split(" ");
		int seqLen = l.length -1;
		int[] obs = new int[l.length];
		String prefixString = new String();
		for (int j = 0; j < seqLen ; j++) {
			obs[j] = Integer.parseInt(l[j+1]);
			prefixString = prefixString + obs[j] + " "; 
		} 
		prefixString = prefixString.trim();
		List<RankStruct> rankList = new ArrayList<RankStruct>();
		for (int j = 0; j < numSymbols ; j++) {
			obs[obs.length-1] = j;
			rankList.add(new RankStruct((j == (numSymbols-1) ? -1 : j), hmm.getSequenceLikelihood(obs)));
		}
		Collections.sort(rankList);
		String rankString = new String();
		for (int index = 4; index >= 0; index --) {
			rankString = rankString + rankList.get(index).getSymbol() + " ";
		} 
		return rankString.trim();
	}

	/**
	 * Returns a new array having (1 - ratio) * a + ratio * b element wise 
	 * @param a
	 * @param b
	 * @param ratio
	 * @return
	 * @throws Exception 
	 */
	public static double[] mixWt(double[] a, double[] b, double ratio) throws Exception {
		if (a.length != b.length)
			throw new Exception("Error in mixing weights");
		double r[] = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			r[i] = (1 - ratio) * a[i] + ratio * b[i];
		}
		return r; 
	}

	public static double[][] mixWt(double[][] a, double[][] b, double ratio) throws Exception {
		if (a.length != b.length || a[0].length != b[0].length)
			throw new Exception("Error in mixing weights");
		double r[][] = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				r[i][j] = (1 - ratio) * a[i][j] + ratio * b[i][j];
			}
		}
		return r; 
	}
}

