from evalsys.prompts import build_user_prompt
from evalsys.rubrics import classify_rubric_line, rubric_for_agent


ANAGRAM_RUBRIC = """
Anagram:
   1. Check if two Strings are Anagrams of each other
   2. Solution 1:
      1. Initialize a map or dictionary or array to store the character frequencies. [1 mark]
      2. For each character in the first string, the corresponding frequency is incremented by 1. [1 mark]
   3. Solution 3:
      1. Sort both the strings. [2 marks]
      2. Return true if both the sorted strings are exactly same. Else, return false [2 marks]
   4. Check for edge cases such as empty strings. [1 mark]
   5. Expected Time Complexity: O(n)
   6. Use clear variable names and comments. [1 mark]
""".strip()


def test_style_gets_only_style_items():
    text = rubric_for_agent("style", ANAGRAM_RUBRIC)
    assert "variable names and comments" in text
    assert "Sort both the strings" not in text
    assert "Time Complexity" not in text


def test_missing_role_rubric_is_empty():
    text = rubric_for_agent("style", "Sort both the strings. [2 marks]")
    assert text == ""


def test_correctness_gets_solution_steps_not_style():
    text = rubric_for_agent("correctness", ANAGRAM_RUBRIC)
    assert "Sort both the strings" in text
    assert "Initialize a map" in text
    assert "variable names and comments" not in text
    assert "Time Complexity" not in text


def test_complexity_and_edge_case_filters():
    assert "O(n)" in rubric_for_agent("complexity", ANAGRAM_RUBRIC)
    assert "edge cases" in rubric_for_agent("edge_cases", ANAGRAM_RUBRIC)
    assert rubric_for_agent("counteragent", ANAGRAM_RUBRIC) == ""


def test_prompt_omits_rubric_when_none_apply():
    prompt = build_user_prompt(
        question="Check anagrams",
        submission="class Solution {}",
        rubric="",
        max_marks=4,
    )
    assert "No official rubric items apply" in prompt
    filled = build_user_prompt(
        question="Check anagrams",
        submission="class Solution {}",
        rubric="Use clear variable names.",
        max_marks=4,
    )
    assert "Official marking rubric for this agent" in filled
    assert "Use clear variable names." in filled


def test_classify_mark_line_as_correctness():
    assert "correctness" in classify_rubric_line("Sort both the strings. [2 marks]")
    assert "style" in classify_rubric_line("Use readable names and comments.")
