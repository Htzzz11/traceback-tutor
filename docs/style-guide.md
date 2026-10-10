# Style guide for explanations and hints

The reader has been programming for about a week. They can read a short function but don't know
words like "type", "index" or "argument" yet. Every target explanation and hint follows these rules.
The five examples in `data/fixtures/examples.jsonl` are the reference style.

## Format

```
Line <N>: <explanation>

Hint: <one question>
```

`<N>` is the line with the bug (`bug_line`). This is not always the line where Python crashed
(`error_line`). For example, a loop's `range` can be wrong on line 3 while the crash happens on line 4.

## The explanation

- **Plain words.** Write "position in the list", not "index"; "the values you pass in", not
  "arguments"; "text", not "string". If a technical word really helps, explain it in a few words
  the first time you use it.
- **Say what went wrong and why.** Describe what the program tried to do and why Python couldn't
  do it. Don't just repeat the error message.
- **Point at the right line.** Talk about the bug line. If the crash happened on a different line,
  mention both and say how they're connected ("the range on line 3 also produces 3, so line 4 asks
  for a position that doesn't exist").
- **Short.** 1–3 sentences and under 80 words.
- You can quote small pieces of the student's own code in backticks, such as `totl` or `age`.
  Never quote code the student hasn't written.

## The hint

- **Exactly one hint, phrased as a question.** It ends with `?`. A short pointer sentence before
  the question is fine ("Compare the name on line 5 with ... Are they spelt exactly the same?").
- **It nudges the student and doesn't do the work.** It tells them where to look or what to
  compare, so they can find the fix themselves.

## Never

- Corrected code, a rewritten line, or a code block of any kind.
- The exact edit, such as "change `totl` to `total`", "add a `:` at the end of line 2" or "use
  `range(len(xs))`".
- More than one hint, or a list of possible fixes.
- Naming the right spelling from Python's `Did you mean: ...?` note. It is the fix. A hint can
  ask the student to read that part of the error instead.

## Bad examples

**Gives the fix** (`str_plus_number`, line 2):
> Line 2: You can't add a string and an int. Use `str(age)` instead.
>
> Hint: Did you try `"Name: " + name + ", age: " + str(age)`?

This uses jargon without explaining it ("string", "int"). Worse, the explanation states the fix and
the hint is the corrected line, so the student has nothing left to work out.

**Too vague** (`misspelt_variable`, line 5):
> Line 5: There is a problem with a variable.
>
> Hint: Can you check your code carefully?

It doesn't say what the problem is or why Python complained. The hint could be attached to any
error, so it doesn't point the student anywhere.

**Wrong line** (`off_by_one_index`, bug on line 3, crash on line 4):
> Line 4: `xs[i]` asks for a position that is too big for the list.
>
> Hint: Is `i` too big here?

Line 4 only shows where the bad number gets used. The mistake that makes `i` too big is the
`range` on line 3, and the explanation never mentions it. The student will start editing line 4.
