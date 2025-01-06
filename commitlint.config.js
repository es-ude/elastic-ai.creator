module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'style', 'refactor', 'revert', 'chore', 'perf','test', 'ci', 'build', 'bump']],
        'header-max-length': [1, 'always', 100]
    }
}
