public class RankStruct implements Comparable<RankStruct> {
	
	private int symbol;
	private double probability;
	
	public RankStruct(int symbol, double probability) {
		this.symbol = symbol;
		this.probability = probability;
	}
	
	public int getSymbol() {
		return symbol;
	}
	public void setSymbol(int symbol) {
		this.symbol = symbol;
	}
	public double getProbability() {
		return probability;
	}
	public void setProbability(double probability) {
		this.probability = probability;
	}
	
	@Override
	public int compareTo(RankStruct r) {
		if (this.getProbability() < r.getProbability()) {
			return -1;
		} else return 1;
	}
	
	@Override
	public String toString() {
		return "RankStruct [symbol=" + symbol + ", probability=" + probability+ "]";
	}	
}
