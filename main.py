# # # tradingagents_v2/main.py
# # from graph.trading_graph import build_graph

# # def analyze(ticker: str):
# #     graph  = build_graph()
# #     result = graph.invoke({"ticker": ticker.upper()})
# #     return result

# # if __name__ == "__main__":
# #     import sys
# #     ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
# #     analyze(ticker)

# from graph.trading_graph import build_graph
# import json

# def analyze(ticker: str):
#     graph = build_graph()
#     result = graph.invoke({"ticker": ticker.upper()})

#     print("\n===== FINAL GRAPH STATE KEYS =====")
#     print(result.keys())

#     print("\n===== LENGTHS =====")
#     print("technical_report length:", len(result.get("technical_report", "")))
#     print("fundamental_report length:", len(result.get("fundamental_report", "")))
#     print("synthesis_output length:", len(result.get("synthesis_output", "")))

#     print("\n===== FINAL SYNTHESIS OUTPUT =====")
#     print(result.get("synthesis_output", ""))

#     return result

# if __name__ == "__main__":
#     import sys
#     ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
#     analyze(ticker)


from graph.trading_graph import build_graph

def analyze(ticker: str):
    graph = build_graph()
    result = graph.invoke({"ticker": ticker.upper()})

    print("\n===== FINAL SYNTHESIS =====\n")
    print(result.get("synthesis_output", ""))

    return result

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    analyze(ticker)