"""축소판 코드 생성 벤치마크. HumanEval 형식을 따랐다.

각 문제는 (프롬프트, 정답, 테스트)로 구성된다. 프롬프트는 모델에 그대로
넣고, 모델이 이어 쓴 본문을 테스트로 채점한다.

`wrong`은 하네스 자체를 검증하기 위한 일부러 틀린 답이다. 채점기가
정답을 통과시키고 오답을 떨어뜨리는지 먼저 확인해야 채점 결과를 믿을 수 있다.
"""

from dataclasses import dataclass


@dataclass
class Problem:
    name: str
    prompt: str
    solution: str  # 정답 본문 (프롬프트에 이어붙임)
    wrong: str  # 일부러 틀린 본문 (하네스 검증용)
    test: str


PROBLEMS: list[Problem] = [
    Problem(
        name="add_two",
        prompt='def add_two(a, b):\n    """두 수를 더해 반환한다."""\n',
        solution="    return a + b\n",
        wrong="    return a - b\n",
        test="assert add_two(2, 3) == 5\nassert add_two(-1, 1) == 0\n",
    ),
    Problem(
        name="is_even",
        prompt='def is_even(n):\n    """n이 짝수면 True."""\n',
        solution="    return n % 2 == 0\n",
        wrong="    return n % 2 == 1\n",
        test="assert is_even(4)\nassert not is_even(7)\nassert is_even(0)\n",
    ),
    Problem(
        name="reverse_string",
        prompt='def reverse_string(s):\n    """문자열을 뒤집어 반환한다."""\n',
        solution="    return s[::-1]\n",
        wrong="    return s\n",
        test="assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n",
    ),
    Problem(
        name="max_of_list",
        prompt='def max_of_list(xs):\n    """리스트의 최댓값을 반환한다. 빈 리스트면 None."""\n',
        solution="    if not xs:\n        return None\n    return max(xs)\n",
        wrong="    return min(xs)\n",
        test="assert max_of_list([1, 5, 3]) == 5\nassert max_of_list([]) is None\n",
    ),
    Problem(
        name="count_vowels",
        prompt='def count_vowels(s):\n    """영어 모음 개수를 센다."""\n',
        solution="    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        wrong="    return len(s)\n",
        test="assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\n",
    ),
    Problem(
        name="fizzbuzz",
        prompt='def fizzbuzz(n):\n    """3의 배수는 Fizz, 5의 배수는 Buzz, 둘 다면 FizzBuzz, 아니면 문자열 숫자."""\n',
        solution=(
            "    if n % 15 == 0:\n        return 'FizzBuzz'\n"
            "    if n % 3 == 0:\n        return 'Fizz'\n"
            "    if n % 5 == 0:\n        return 'Buzz'\n"
            "    return str(n)\n"
        ),
        wrong="    return str(n)\n",
        test=(
            "assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(5) == 'Buzz'\n"
            "assert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(7) == '7'\n"
        ),
    ),
    Problem(
        name="sum_list",
        prompt='def sum_list(xs):\n    """리스트 원소의 합."""\n',
        solution="    total = 0\n    for x in xs:\n        total += x\n    return total\n",
        wrong="    return len(xs)\n",
        test="assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\n",
    ),
    Problem(
        name="unique_sorted",
        prompt='def unique_sorted(xs):\n    """중복을 없애고 정렬한 리스트를 반환한다."""\n',
        solution="    return sorted(set(xs))\n",
        wrong="    return xs\n",
        test="assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]\nassert unique_sorted([]) == []\n",
    ),
    Problem(
        name="factorial",
        prompt='def factorial(n):\n    """n의 팩토리얼. n이 0이면 1."""\n',
        solution=(
            "    result = 1\n    for i in range(2, n + 1):\n"
            "        result *= i\n    return result\n"
        ),
        wrong="    return n\n",
        test="assert factorial(0) == 1\nassert factorial(5) == 120\n",
    ),
    Problem(
        name="word_count",
        prompt='def word_count(s):\n    """공백으로 나눈 단어 개수."""\n',
        solution="    return len(s.split())\n",
        wrong="    return len(s)\n",
        test="assert word_count('a b c') == 3\nassert word_count('') == 0\n",
    ),
]
