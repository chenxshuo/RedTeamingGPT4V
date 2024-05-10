# -*- coding: utf-8 -*-

"""Some methods for pre- and post-process on the input to the LM and generation of the LM."""

import logging
import base64

logger = logging.getLogger(__name__)


def base64_encode_input(input_string):
    encoded_bytes = base64.b64encode(input_string.encode("utf-8"))
    encoded_string = str(encoded_bytes, "utf-8")
    return encoded_string


def base64_decode_output(input_string):
    try:
        # sometimes, the generation result is not base64 encoded
        # Decode the Base64 encoded string
        decoded_bytes = base64.b64decode(input_string)
        # Convert bytes to string
        decoded_string = decoded_bytes.decode("utf-8")
        return decoded_string
    except Exception as e:
        return input_string + f"\n # Error: {e}"


def remove_vowels(input_string):
    vowels = "aeiouAEIOU"
    return "".join(char for char in input_string if char not in vowels)


def leetspeak_obfuscation(input_string):
    transformations = {
        "a": "@",
        "e": "3",
        "i": "!",
        "o": "0",
        "are": "r",
        "be": "b",
        "s": "$",
        "t": "7",
        "l": "1",
        "g": "9",
        "z": "2",
        "q": "9",
        "u": "|_|",
        "you": "u",
        "for": "4",
        "and": "&",
    }

    for key, value in transformations.items():
        input_string = input_string.replace(key, value)
    return input_string


def rot13(text):
    result = ""
    for char in text:
        if char.isalpha():
            shifted = ord(char) + 13
            if char.islower():
                if shifted > ord("z"):
                    shifted -= 26
            elif char.isupper():
                if shifted > ord("Z"):
                    shifted -= 26
            result += chr(shifted)
        else:
            result += char
    return result


def auto_payload_splitting(text):
    raise NotImplementedError


INPUT_FUNCTON_DICT = {
    "base64": base64_encode_input,
    "rot13": rot13,
    "leetspeak_obfuscation": leetspeak_obfuscation,
    "remove_vowels": remove_vowels,
    "auto_payload_splitting": auto_payload_splitting,
}

OUTPUT_FUNCTION_DICT = {
    "base64": base64_decode_output,
}
