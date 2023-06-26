import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


class HandEngineeredExtractor:
    def __init__(self):
        rules = self._load_rules()
        self._regex = self._compile_rules(rules)

    def _load_rules(self):
        return {
            "eng_prefix": "^[anti|de|dis|en|fore|in|mid|mis|non|over|pre|re|semi|sub|super|trans|un|under].*",
            "eng_sufix": ".*[able|ible|al|ial|ed|en|er|er|est|ful|ic|ing|ion|ty|ive|less|ment|ness|ous|s|y]$",
            "cro_prefix": "^[bez|be|bes|beš|beza|iz|is|iza|među|na|nad|nat|ne|o|ob|op|od|ot|po|pod|pot|pra|pre|pred|pri|pro|raz|ras|raš|raza|s|sa|z|su|u|uz].*",
            "cro_sufix": ".+[skoga|skima|skom|skoj|skog|skim|skih|noga|sku|sko|ski|ske|ska|nom|noj|nog|nim|nih|na|nu|no|ni|ne|anjima|enjima|stvima|ovima|evima|enoga|anoga|anjem|enjem|stvom|stvo|stva|stvu|anje|enje|anja|enja|enom|enoj|enog|enim|enih|anom|anoj|anog|anim|anih|e|no|ano|ovi|ova|oga|ima|evi|eva|ove|eve|enu|eni|ene|anu|ani|ane|ena|ana|ama|om|og|im|ih|em|oj|u|o|i|e|a]&",
            "double_letters": "(.)\\1",
            "eng_letters": "[q|w|y|x|]",
            "cro_letters": "[č|ć|đ|š|ž]"
        }

    def _compile_rules(self, rules):
        return {name: self._compile_re(rule) for name, rule in rules.items()}

    def _compile_re(self, rule):
        return re.compile(r'{}'.format(rule))

    def word_to_vec(self, word):
        assert type(word) == str
        return self._convert_to_int_array(
                (len(word) - self._mean_len) / self._std,
                self._containes_double_letters(word),
                self._containes_dash(word),
                self._containes_eng_letters(word),
                self._containes_cro_letters(word),
                self._containes_eng_prefix(word),
                self._containes_cro_prefix(word),
                self._containes_eng_sufix(word),
                self._containes_cro_sufix(word)
        )

    def fit_transform(self, words, _):
        self.fit(words)
        return self.transform(words)

    def fit(self, words):
        import statistics
        sizes = [len(w) for w in words]
        self._mean_len = statistics.mean(sizes)
        self._std = statistics.stdev(sizes)

    def transform(self, words):
        return np.array([self.word_to_vec(word) for word in words])

    def _containes_double_letters(self, word):
        return self._search(self._regex['double_letters'], word)

    def _containes_dash(self, word):
        return '-' in word

    def _containes_eng_letters(self, word):
        return self._search(self._regex['eng_letters'], word)

    def _containes_cro_letters(self, word):
        return self._search(self._regex['cro_letters'], word)

    def _containes_eng_prefix(self, word):
        return self._search(self._regex['eng_prefix'], word)

    def _containes_cro_prefix(self, word):
        return self._search(self._regex['cro_prefix'], word)

    def _containes_eng_sufix(self, word):
        return self._search(self._regex['eng_sufix'], word)

    def _containes_cro_sufix(self, word):
        return self._search(self._regex['cro_sufix'], word)

    def _search(self, regex, word):
        return re.search(regex, word) is not None

    def _convert_to_int_array(self, *args):
        return [int(v) for v in args]


class NgramAndHandEngineeredFeatuerExtractor:
    def __init__(self, ngram_range):
        self._ngrams = CountVectorizer(
            analyzer="char", ngram_range=ngram_range)
        self._he = HandEngineeredExtractor()

    def fit_transform(self, words, _):
        self.fit(words)
        return self.transform(words)

    def fit(self, words):
        self._he.fit(words)
        self._ngrams.fit(words)

    def transform(self, words):
        he = self._he.transform(words)
        ngrams = self._ngrams.transform(words)
        return hstack((ngrams, he))
