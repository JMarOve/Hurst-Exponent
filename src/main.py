from Hurst.hurstexponent import *



if __name__ == "__main__":
    analyzer = HurstAnalyzer(min_window=10)

    try:
        analyzer.analyze_financial(
            symbol='LLOY.L',
            start_date='2020-01-01'
        )

    except ValueError as e:
        print(f"Analysis failed: {str(e)}")