from pathlib import Path

from evalsys.dataset import load_samples, parse_marks


def test_parse_marks_fraction_and_excel_date():
    assert parse_marks("0/4") == (0.0, 4.0)
    assert parse_marks("3/4") == (3.0, 4.0)
    assert parse_marks("04-Apr") == (4.0, 4.0)
    assert parse_marks("02-Jun") == (2.0, 6.0)
    assert parse_marks("11-Dec") == (11.0, 12.0)
    assert parse_marks("08-Nov") == (8.0, 11.0)
    assert parse_marks(4) == (None, 4.0)
    assert parse_marks(None) == (None, None)


def test_load_csv_reads_problem_solution_and_max_marks(tmp_path: Path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "question,rubric,submission,category,total marks\n"
        '"Check anagrams","Sort both strings. [2 marks]","class Solution {}",Compilation_Error,04-Apr\n'
        '"Sum two numbers","","int main(){}",Wrong,0/5\n',
        encoding="utf-8",
    )
    samples = load_samples(path)
    assert len(samples) == 2
    assert samples[0].question.startswith("Check anagrams")
    assert "class Solution" in samples[0].submission
    assert samples[0].category == "Compilation_Error"
    assert samples[0].max_marks == 4.0
    assert samples[0].awarded_marks == 4.0
    assert samples[0].ground_truth == "4/4"
    assert samples[0].rubric == "Sort both strings. [2 marks]"
    assert samples[1].max_marks == 5.0
    assert samples[1].awarded_marks == 0.0
    assert samples[1].ground_truth == "0/5"
