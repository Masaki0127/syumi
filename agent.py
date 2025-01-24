from typing import Literal, Optional
from datetime import date
import json

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, Command, START, END
from langgraph.prebuilt import MessagesState, create_react_agent
from langgraph.prebuilt.react import tool

###############################################################################
# 1. 状態定義
###############################################################################

class QueryState(MessagesState):
    """
    LangGraphで各ノード間をやり取りする際に共有する状態。
    - query: 質問や途中生成される回答を格納
    - agent2_judgment: Agent2のYes/No
    - agent4_judgment: Agent4のYes/No
    - debug_info: 全体の処理履歴などを文字列で蓄積
    """
    query: str = ""
    agent2_judgment: Optional[Literal["Yes", "No"]] = None
    agent4_judgment: Optional[Literal["Yes", "No"]] = None
    debug_info: list = []


###############################################################################
# 2. 各Agentのツール定義(ReAct用)
#
#   - "@tool" デコレータを使うと、ReActエージェントが呼び出せるツールとして自動登録されます
#   - docstring(Triple-quoted) がそのままLLMに渡るため、なるべく「入力/出力形式」を明記するのが望ましい
###############################################################################

##
# Agent1向けツール
##
@tool
def agent1_tool1_check_date_in_query(input_text: str) -> str:
    """
    This tool checks if the input_text contains keywords like '最新の' or '現在の'.
    If found, return today's date as a string. Otherwise return an empty string.
    Output format: "YYYY-MM-DD" or "" (empty string).
    """
    keywords = ["最新の", "現在の"]
    if any(k in input_text for k in keywords):
        return str(date.today())
    return ""

##
# Agent2向けツール
##
@tool
def agent2_tool1_check_dictionary(input_text: str) -> str:
    """
    This tool checks if '特殊用語' is included in the input_text.
    Output: "Yes" if input_text contains '特殊用語', otherwise "No".
    """
    if "特殊用語" in input_text:
        return "Yes"
    return "No"

##
# Agent3向けツール
##
@tool
def agent3_tool1_search_document(query: str) -> str:
    """
    This tool searches documents for 'query' and returns a dummy string result.
    Output format: a string with retrieved info.
    """
    return f"[Document] Found info about '{query}'"

@tool
def agent3_tool2_search_slack(query: str) -> str:
    """
    This tool searches Slack for 'query' and returns a dummy string result.
    Output format: a string with retrieved info.
    """
    return f"[Slack] Found info about '{query}'"

@tool
def agent3_tool3_search_github_issue(query: str) -> str:
    """
    This tool searches GitHub issues for 'query' and returns a dummy string result.
    Output format: a string with retrieved info.
    """
    return f"[GitHubIssue] Found info about '{query}'"

@tool
def agent3_tool4_search_internet(term: str) -> str:
    """
    This tool searches the Internet for 'term' and returns a dummy string result.
    Output format: a string with retrieved info.
    """
    return f"[Internet] Found info about '{term}'"

##
# Agent4向けツール
##
@tool
def agent4_tool1_check_unanswered(input_json: str) -> str:
    """
    This tool checks if there is any unanswered portion in the query vs. answer.
    input_json = { "query": "...", "answer_candidate": "..." }
    Output: "Yes" if all answered, "No" if something is missing.
    """
    data = json.loads(input_json)
    query = data["query"]
    answer_candidate = data["answer_candidate"]

    # ダミー判定: "詳しく" が query にあるが answer_candidate に無い→ No
    if "詳しく" in query and "詳しく" not in answer_candidate:
        return "No"
    return "Yes"

@tool
def agent4_tool2_check_technical_words(input_json: str) -> str:
    """
    This tool checks if the answer_candidate is too technical or not.
    input_json = { "query": "...", "answer_candidate": "..." }
    Output: "Yes" if the answer is easy enough, "No" if it's too difficult.
    """
    data = json.loads(input_json)
    # ここでは雑に "専門用語" が回答に入っていればNoとする
    if "専門用語" in data["answer_candidate"]:
        return "No"
    return "Yes"


###############################################################################
# 3. 各Agent(= ReActエージェント)の定義: create_react_agent
###############################################################################
model = ChatOpenAI(temperature=0.0)

# Agent1: agent1_tool1_check_date_in_query
#   → "最新の" や "現在の" が入っているか調べ、あれば日付を返すツール
#   → ここではあまり複雑でないため1つのツールだけ割り当て。
agent1_graph = create_react_agent(
    llm = model,
    tools = [agent1_tool1_check_date_in_query],
    name = "agent1_react",
    description = """
    Agent1: Given the user's query, clarify or regenerate the query to make it easier to answer.
    Tools:
      - agent1_tool1_check_date_in_query
    The final answer from this agent should describe how (or if) you appended date info to the query.
    If no date needed, just say "No date appended.".
    """
)

# Agent2: agent2_tool1_check_dictionary
#   → "特殊用語" があるかどうかをチェック。Yes/Noを最終回答として返す
agent2_graph = create_react_agent(
    llm = model,
    tools = [agent2_tool1_check_dictionary],
    name = "agent2_react",
    description = """
    Agent2: Judge whether the query contains unknown technical words, returning "Yes" or "No".
    Tools:
      - agent2_tool1_check_dictionary
    The final answer from this agent must be exactly "Yes" or "No".
    """
)

# Agent3: 4つのツールを用いて不足情報を収集する
agent3_graph = create_react_agent(
    llm = model,
    tools = [
        agent3_tool1_search_document,
        agent3_tool2_search_slack,
        agent3_tool3_search_github_issue,
        agent3_tool4_search_internet
    ],
    name = "agent3_react",
    description = """
    Agent3: Collect missing info from multiple sources (document, slack, github issue, internet).
    You can call any or all tools as needed.
    The final answer from this agent should be a summary of what you've found.
    """
)

# Agent4: 作られた回答が分かりやすいかチェック ("Yes"/"No")
#   → agent4_tool1_check_unanswered
#   → agent4_tool2_check_technical_words
agent4_graph = create_react_agent(
    llm = model,
    tools = [agent4_tool1_check_unanswered, agent4_tool2_check_technical_words],
    name = "agent4_react",
    description = """
    Agent4: Evaluate the final answer to see if it fully addresses the query and is not too technical.
    Use these two tools and combine their results.
    The final answer must be "Yes" if both checks are good, or "No" otherwise.
    - agent4_tool1_check_unanswered
    - agent4_tool2_check_technical_words
    You can call each tool with a JSON input { "query": ..., "answer_candidate": ... }.
    Then combine their results. If both are "Yes", you output "Yes". If either is "No", output "No".
    """
)


###############################################################################
# 4. 各エージェント呼び出しをラップするノード関数
#
#   LangGraph でサブグラフ(= ReAct agent)を扱う場合は、直接 add_node(agent1_graph)
#   するのではなく、下記のように "ラッパ関数" を作って Command を返すと
#   グラフの可視化や制御フローがより扱いやすくなります。
###############################################################################

def call_agent1(state: QueryState) -> Command[Literal["agent2"]]:
    """
    Agent1サブグラフを実行 → 生成物を state.query に反映し、agent2へ進む
    """
    # ReActエージェントに渡すプロンプトとして、MessagesState["messages"] を使う方法もありますが
    # ここでは簡単に指示文の末尾に state.query を与えるだけにします。
    instructions = f"User Query: {state.query}\nPlease finalize your answer now."
    result = agent1_graph.invoke({"messages": [{"role": "user", "content": instructions}]})
    # result は ToolMessage, AIMessage, etc. の配列が返ってくる
    # 最終回答(AIMessage)を抽出
    final_ans = result[-1]["content"]

    # 例: Agent1の回答は「日付をappendしたかどうか」なので、それを state.query に取り込み
    # 実際の使い方は用途に合わせて変更
    state.query = f"{state.query}\n[Agent1 says]: {final_ans}"
    state.debug_info.append(f"[Agent1] {final_ans}")

    return Command(
        goto="agent2",
        update={"query": state.query, "debug_info": state.debug_info}
    )


def call_agent2(state: QueryState) -> Command[Literal["router1"]]:
    """
    Agent2サブグラフを実行 → 最終回答("Yes"/"No")を parse して state.agent2_judgment に格納
    """
    instructions = f"User Query: {state.query}\nReturn final answer as 'Yes' or 'No'."
    result = agent2_graph.invoke({"messages": [{"role": "user", "content": instructions}]})
    final_ans = result[-1]["content"].strip()

    # Agent2の結果を state.agent2_judgment にセット
    if final_ans not in ["Yes", "No"]:
        # 万一 "Yes"/"No" 以外が来た場合のfallback
        final_ans = "No"
    state.agent2_judgment = final_ans
    state.debug_info.append(f"[Agent2] judged = {final_ans}")

    return Command(
        goto="router1",
        update={
            "agent2_judgment": state.agent2_judgment,
            "debug_info": state.debug_info
        }
    )


def call_agent3(state: QueryState) -> Command[Literal["prompt1"]]:
    """
    Agent3サブグラフを実行 → 取得した追加情報を state.query に付加して prompt1へ進む
    """
    instructions = (
        "We need more info for the user query. Please gather from documents, Slack, GitHub issues, internet. "
        f"User Query: {state.query}\nFinally, output a summary of what you found."
    )
    result = agent3_graph.invoke({"messages": [{"role": "user", "content": instructions}]})
    final_ans = result[-1]["content"]

    # 取得した情報を query に付加する例
    state.query = f"{state.query}\n[Agent3 additional info]:\n{final_ans}"
    state.debug_info.append(f"[Agent3] {final_ans}")

    return Command(
        goto="prompt1",
        update={"query": state.query, "debug_info": state.debug_info}
    )


def call_agent4(state: QueryState) -> Command[Literal["router2"]]:
    """
    Agent4サブグラフを実行 → "Yes" or "No" で回答が十分かどうか判断
    """
    # Agent4は2つのツールを呼び出し、それぞれの結果をまとめて最終回答("Yes"/"No")を返す想定
    # ここでは query(=ユーザの質問) と answer_candidate(= 現在の回答) をJSONで渡すように促す
    data_json = json.dumps({"query": state.query, "answer_candidate": state.query})
    instructions = (
        "Check the answer thoroughly using your tools to see if it's fully answered and not too technical.\n"
        f"JSON to pass to each tool: {data_json}\n"
        "Finally, output 'Yes' if it's fully acceptable, else 'No'."
    )
    result = agent4_graph.invoke({"messages": [{"role": "user", "content": instructions}]})
    final_ans = result[-1]["content"].strip()

    if final_ans not in ["Yes", "No"]:
        final_ans = "No"

    state.agent4_judgment = final_ans
    state.debug_info.append(f"[Agent4] judged = {final_ans}")

    return Command(
        goto="router2",
        update={
            "agent4_judgment": state.agent4_judgment,
            "debug_info": state.debug_info
        }
    )


###############################################################################
# 5. Promptノード実装 (普通のLLM呼び出し; 必要ならReAct化してもOK)
###############################################################################

def prompt1(state: QueryState) -> Command[Literal["agent4"]]:
    """
    Prompt1: 得られた情報をもとに一旦回答を生成し、次にAgent4へ渡す。
    """
    llm = ChatOpenAI()
    prompt_text = f"""
あなたはユーザに回答するアシスタントです。
[User Query & Collected Info]: {state.query}

この質問に対して、一度回答案をまとめてください。
"""
    response = llm.invoke(prompt_text)
    answer = response.content.strip()

    state.query = f"{state.query}\n[Prompt1's answer draft]:\n{answer}"
    state.debug_info.append(f"[Prompt1] generated answer draft:\n{answer}")

    return Command(
        goto="agent4",
        update={"query": state.query, "debug_info": state.debug_info}
    )


def prompt2(state: QueryState) -> Command[Literal["agent2"]]:
    """
    Prompt2: Agent4がNoと判断した場合に再度回答をブラッシュアップする
    """
    llm = ChatOpenAI()
    prompt_text = f"""
あなたは回答をブラッシュアップするアシスタントです。
前回の回答案:
{state.query}

まだ回答が不十分 or 専門的すぎると指摘されました。
より分かりやすく、より詳細に説明する回答を再度作成してください。
"""
    response = llm.invoke(prompt_text)
    refined_answer = response.content.strip()

    state.query = f"{state.query}\n[Prompt2 refined answer]:\n{refined_answer}"
    state.debug_info.append(f"[Prompt2] refined answer:\n{refined_answer}")

    # ブラッシュアップ後、Agent2へ戻る (再度専門用語チェックなど)
    return Command(
        goto="agent2",
        update={"query": state.query, "debug_info": state.debug_info}
    )


###############################################################################
# 6. ルーターノード(Yes/No分岐)
###############################################################################

def router1(state: QueryState) -> Command[Literal["prompt1", "agent3"]]:
    """
    Router1:
      Agent2の判定(Yes/No)により遷移先を分岐。
      - Yes -> prompt1
      - No  -> agent3
    """
    if state.agent2_judgment == "Yes":
        goto_node = "prompt1"
    else:
        goto_node = "agent3"

    state.debug_info.append(f"[Router1] Agent2 = {state.agent2_judgment} -> {goto_node}")
    return Command(goto=goto_node, update={"debug_info": state.debug_info})


def router2(state: QueryState) -> Command[Literal["prompt2", END]]:
    """
    Router2:
      Agent4の判定(Yes/No)により遷移先を分岐。
      - Yes -> END (回答を確定して終了)
      - No  -> prompt2
    """
    if state.agent4_judgment == "Yes":
        state.debug_info.append("[Router2] Agent4=Yes => END")
        return Command(goto=END, update={"debug_info": state.debug_info})
    else:
        state.debug_info.append("[Router2] Agent4=No => prompt2")
        return Command(goto="prompt2", update={"debug_info": state.debug_info})


###############################################################################
# 7. グラフ組み立て
###############################################################################

def build_graph():
    builder = StateGraph(QueryState)

    # 各ノード登録 (Agent1～Agent4は ReActサブグラフを呼び出す "ラッパ" 関数で登録)
    builder.add_node(call_agent1,  name="agent1")
    builder.add_node(call_agent2,  name="agent2")
    builder.add_node(router1,      name="router1")
    builder.add_node(call_agent3,  name="agent3")
    builder.add_node(prompt1,      name="prompt1")
    builder.add_node(call_agent4,  name="agent4")
    builder.add_node(router2,      name="router2")
    builder.add_node(prompt2,      name="prompt2")

    # エッジ定義
    builder.add_edge(START, "agent1")     # query => Agent1
    builder.add_edge("agent1", "agent2")  # Agent1 => Agent2
    builder.add_edge("agent2", "router1") # Agent2 => Router1
    builder.add_edge("router1", "prompt1")
    builder.add_edge("router1", "agent3")
    builder.add_edge("agent3", "prompt1") # Agent3 => Prompt1
    builder.add_edge("prompt1", "agent4") # Prompt1 => Agent4
    builder.add_edge("agent4", "router2") # Agent4 => Router2
    builder.add_edge("router2", END)
    builder.add_edge("router2", "prompt2")
    builder.add_edge("prompt2", "agent2")

    # これで構造:
    # query -> agent1 -> agent2 -> router1 -> [Yes->prompt1, No->agent3] -> prompt1 -> agent4 -> router2 -> [Yes->END, No->prompt2->agent2->...]
    return builder.compile()


###############################################################################
# 8. 実行例
###############################################################################

if __name__ == "__main__":
    graph = build_graph()

    # テスト用State
    test_state = QueryState(
        query="特殊用語について詳しく知りたいです。最新の情報も教えてください。"
    )

    final_state = graph.run(test_state)

    print("========== [Execution Finished] ==========")
    print("Final Query State:")
    print(final_state.query)
    print("\n--- Debug Info ---")
    for line in final_state.debug_info:
        print(line)
