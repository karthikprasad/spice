import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.*;
import java.util.Random;

public class HMM {

	/** number of hidden states */
	private int latentStates;

	/** number of observable states */
	private int emissionStates;

	/** initial distribution */
	private double pi[];

	/** state transition matrix */
	private double transMat[][];

	/** state emission matrix */
	private double emissionMat[][];

	private double tolerance;

	private int maxiter;

	private boolean converged = false;

	// Getters and setters
	public boolean getConverged() {
		return this.converged;
	}

	public double[] getPi() {
		return pi;
	}

	public void setPi(double[] pi) {
		this.pi = pi;
	}

	public double[][] getA() {
		return transMat;
	}

	public void setA(double[][] a) {
		this.transMat = a;
	}

	public double[][] getB() {
		return emissionMat;
	}

	public void setB(double[][] b) {
		this.emissionMat = b;
	}

	public HMM(int latentStates, int emissionStates, int maxiter, double tolerance) {
		this.latentStates = latentStates;
		this.emissionStates = emissionStates;
		this.tolerance = tolerance;
		this.maxiter = maxiter;

		pi = getRand(latentStates);
		transMat = getRand(latentStates, latentStates);
		emissionMat = getRand(latentStates, emissionStates);
	}

	public void train(int[] inputSequence) throws Exception {
		double[][] forwardMatrix;
		double[][] backwardMatrix;
		for (int itr = 0; itr < maxiter; itr++) {
			double tmpInitial[] = new double[latentStates];
			double tmpTransition[][] = new double[latentStates][latentStates];
			double tmpEmission[][] = new double[latentStates][emissionStates];

			/* Find Forward and Backward probabilities */
			forwardMatrix = findForwardMatrix(inputSequence);
			//printMatrix(forwardMatrix);
			backwardMatrix = findBackWardMatrix(inputSequence);
			//System.out.println("Backward");


			/* re-estimation of initial state probabilities */
			for (int i = 0; i < latentStates; i++)
				tmpInitial[i] = getGamma(i, 0, inputSequence, forwardMatrix, backwardMatrix);
			//System.out.println("pi1 values");
			//for (double d : pi)
				//System.out.println(d);

			if (true)
				//throw new Exception("Bros before hoes");

				/* Estimate transition Matrix */ 
				for (int i = 0; i < latentStates; i++) {
					for (int j = 0; j < latentStates; j++) {
						double sumTop = 0;
						double sumBottom = 0;
						for (int t = 0; t <= inputSequence.length - 1; t++) {
							sumTop += p(t, i, j, inputSequence, forwardMatrix, backwardMatrix);
							sumBottom += getGamma(i, t, inputSequence, forwardMatrix, backwardMatrix);
						}
						tmpTransition[i][j] = divide(sumTop, sumBottom);
					}
				}

			/* Estimate emission Matrix */
			for (int i = 0; i < latentStates; i++) {
				for (int k = 0; k < emissionStates; k++) {
					double sumTop = 0;
					double sumBottom = 0;
					for (int t = 0; t <= inputSequence.length - 1; t++) {
						double g = getGamma(i, t, inputSequence, forwardMatrix, backwardMatrix);
						sumTop += g * (k == inputSequence[t] ? 1 : 0);
						sumBottom += g;
					}
					tmpEmission[i][k] = divide(sumTop, sumBottom);
				}
			}
			if(findDifference(transMat, tmpTransition, emissionMat, tmpEmission) < tolerance) {
				this.converged = true;
				//print();
				System.out.println("Batch converged in " + itr + " steps.");
				break;
			}
			pi = tmpInitial;
			transMat = tmpTransition;
			emissionMat = tmpEmission;
		}
	}


	private double findDifference(double[][] a2, double[][] a1, double[][] b2, double[][] b1) {
		double diff = 0;
		for (int i = 0; i < latentStates; i++) {
			for (int j = 0; j < latentStates; j++) {
				diff = diff + Math.abs(a1[i][j]-a2[i][j]);
			}
			for (int j = 0; j < emissionStates; j++) {
				diff = diff + Math.abs(b1[i][j]-b2[i][j]);
			}
		}
		return diff;
	}

	/** calculation of Forward-Variables f(i,t) for state i at time
      t for output sequence O with the current HMM parameters
      @param o the output sequence O
      @return an array f(i,t) over states and times, containing
              the Forward-variables. 
	 */
	public double[][] findForwardMatrix(int[] o) {
		int T = o.length;
		double[][] fwd = new double[latentStates][T];

		/* initialization (time 0) */
		for (int i = 0; i < latentStates; i++)
			fwd[i][0] = pi[i] * emissionMat[i][o[0]];

		/* induction */
		for (int t = 0; t <= T-2; t++) {
			for (int j = 0; j < latentStates; j++) {
				fwd[j][t+1] = 0;
				for (int i = 0; i < latentStates; i++)
					fwd[j][t+1] += (fwd[i][t] * transMat[i][j]);
				fwd[j][t+1] *= emissionMat[j][o[t+1]];
				//System.out.println(fwd[j][t+1]);
			}
		}

		return fwd;
	}

	public double getSequenceLikelihood(int[] o) {
		double[][] fwd = findForwardMatrix(o);
		double prob = 0;
		for(int i = 0; i < latentStates; i++) {
			prob += fwd[i][o.length-1];
		}
		return prob;
	}

	public double[][] findBackWardMatrix(int[] o) {
		double[][] bwd = new double[latentStates][o.length];

		/* initialization (time 0) */
		for (int i = 0; i < latentStates; i++)
			bwd[i][o.length-1] = 1;

		/* induction */
		for (int t = o.length - 2; t >= 0; t--) {
			for (int i = 0; i < latentStates; i++) {
				bwd[i][t] = 0;
				for (int j = 0; j < latentStates; j++)
					bwd[i][t] += (bwd[j][t+1] * transMat[i][j] * emissionMat[j][o[t+1]]);
			}
		}
		return bwd;
	}

	public double p(int t, int i, int j, int[] o, double[][] fwd, double[][] bwd) {
		double num;
		if (t == o.length - 1)
			num = fwd[i][t] * transMat[i][j];
		else
			num = fwd[i][t] * transMat[i][j] * emissionMat[j][o[t+1]] * bwd[j][t+1];

		double denom = 0;

		for (int k = 0; k < latentStates; k++)
			denom += (fwd[k][t] * bwd[k][t]);

		return divide(num, denom);
	}

	public double getGamma(int i, int t, int[] o, double[][] fwd, double[][] bwd) {
		double numerator = fwd[i][t] * bwd[i][t];
		double denominator = 0;

		for (int j = 0; j < latentStates; j++)
			denominator += fwd[j][t] * bwd[j][t];

		return divide(numerator, denominator);
	}

	public void print() {
		DecimalFormat fmt = new DecimalFormat();
		fmt.setMinimumFractionDigits(5);
		fmt.setMaximumFractionDigits(5);

		for (int i = 0; i < latentStates; i++)
			System.out.println("pi(" + i + ") = " + fmt.format(pi[i]));
		System.out.println();

		for (int i = 0; i < latentStates; i++) {
			for (int j = 0; j < latentStates; j++)
				System.out.print("a(" + i + "," + j + ") = " + 
						fmt.format(transMat[i][j]) + "  ");
			System.out.println();
		}

		System.out.println();
		for (int i = 0; i < latentStates; i++) {
			for (int k = 0; k < emissionStates; k++)
				System.out.print("b(" + i + "," + k + ") = " + 
						fmt.format(emissionMat[i][k]) + "  ");
			System.out.println();
		}
	}

	/** divides two doubles. 0 / 0 = 0! */
	public double divide(double n, double d) {
		if (n == 0)
			return 0;
		else
			return n / d;
	}

	public double[] getRand(int n) {
		Random rand = new Random(); 
		double r[] = new double[n];
		double sum = 0;
		for (int i = 0; i < n; i++) {
			r[i] = rand.nextInt(100);
			sum += r[i];
		}
		for (int i = 0; i < n; i++) {
			r[i] = r[i] / sum;
		}
		return r; 
	}

	public double[][] getRand(int n, int m) {
		Random rand = new Random(); 
		double r[][] = new double[n][m];
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int j = 0; j < m; j++) {
				r[i][j] = rand.nextInt(100);
				sum += r[i][j];
			}
			for (int j = 0; j < m; j++) {
				r[i][j] = r[i][j] / sum;
			}
		}
		return r; 
	}

	public void store(String fileName) throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(fileName, "UTF-8");
		writer.println("### The number of states followed by alphabet size, followed by pi(initial probability), a(transition matrix) and b(emission matrix)");
		writer.println(latentStates + " " + emissionStates);
		DecimalFormat fmt = new DecimalFormat();
		fmt.setMinimumFractionDigits(6);
		fmt.setMaximumFractionDigits(6);

		for (int i = 0; i < latentStates; i++)
			writer.println(fmt.format(pi[i]));
		for (int i = 0; i < latentStates; i++) {
			for (int j = 0; j < latentStates; j++)
				writer.print(fmt.format(transMat[i][j]) + " ");
			writer.println();
		}
		for (int i = 0; i < latentStates; i++) {
			for (int k = 0; k < emissionStates; k++)
				writer.print(fmt.format(emissionMat[i][k]) + " ");
			writer.println();
		}
		writer.close();
	}

	public void printMatrix(double[][] a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++)
				System.out.print(a[i][j] + "  ");
			System.out.println();
		}
	}

}
