module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'doc', 'style', 'refactor', 'test', 'revert', 'ci', 'wip']],
        'subject-empty': [0, "never"],
        'type-empty': [0, "never"]
    }
}
