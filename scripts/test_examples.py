#!/usr/bin/env python3
"""Run representative and edge cases directly against the documented C++ code."""
from pathlib import Path
import re
import subprocess
import tempfile
from check_cpp import ROOT, HEADERS, examples

# Each method case also runs against all documented variants of that method.
CASES = {
 ('数组','twoSum'): 'vector<int> a{2,7,11,15}; auto r=s.twoSum(a,9); assert(r==vector<int>({0,1}) || r==vector<int>({1,2}));',
 ('数组','isAnagram'): 'assert(s.isAnagram("anagram","nagaram")); assert(!s.isAnagram("rat","car"));',
 ('数组','groupAnagrams'): 'vector<string> a{"eat","tea","tan","ate","nat","bat"}; auto r=s.groupAnagrams(a); vector<int> sizes; for(auto& g:r) sizes.push_back(g.size()); sort(sizes.begin(),sizes.end()); assert(sizes==vector<int>({1,2,3}));',
 ('数组','longestConsecutive'): 'vector<int> a{100,4,200,1,3,2,2}; assert(s.longestConsecutive(a)==4); a={INT_MIN,INT_MIN+1,INT_MAX}; assert(s.longestConsecutive(a)==2);',
 ('数组','firstMissingPositive'): 'vector<int> a{3,4,-1,1}; assert(s.firstMissingPositive(a)==2); a={1,2,0}; assert(s.firstMissingPositive(a)==3);',
 ('数组','threeSum'): 'vector<int> a{-1,0,1,2,-1,-4}; auto r=s.threeSum(a); sort(r.begin(),r.end()); assert(r==vector<vector<int>>({{-1,-1,2},{-1,0,1}})); a={0,0,0,0}; assert(s.threeSum(a)==vector<vector<int>>({{0,0,0}}));',
 ('数组','maxArea'): 'vector<int> a{1,8,6,2,5,4,8,3,7}; assert(s.maxArea(a)==49);',
 ('数组','trap'): 'vector<int> a{0,1,0,2,1,0,1,3,2,1,2,1}; assert(s.trap(a)==6); a={}; assert(s.trap(a)==0);',
 ('数组','sortColors'): 'vector<int> a{2,0,2,1,1,0}; s.sortColors(a); assert(a==vector<int>({0,0,1,1,2,2})); a={2}; s.sortColors(a); assert(a[0]==2);',
 ('数组','nextPermutation'): 'vector<int> a{1,2,3}; s.nextPermutation(a); assert(a==vector<int>({1,3,2})); a={3,2,1}; s.nextPermutation(a); assert(a==vector<int>({1,2,3}));',
 ('数组','moveZeroes'): 'vector<int> a{0,1,0,3,12}; s.moveZeroes(a); assert(a==vector<int>({1,3,12,0,0}));',
 ('数组','findDuplicate'): 'vector<int> a{1,3,4,2,2}; assert(s.findDuplicate(a)==2);',
 ('数组','minSubArrayLen'): 'vector<int> a{2,3,1,2,4,3}; assert(s.minSubArrayLen(7,a)==2); assert(s.minSubArrayLen(100,a)==0);',
 ('数组','numSubarrayProductLessThanK'): 'vector<int> a{10,5,2,6}; assert(s.numSubarrayProductLessThanK(a,100)==8); assert(s.numSubarrayProductLessThanK(a,1)==0);',
 ('数组','lengthOfLongestSubstring'): 'assert(s.lengthOfLongestSubstring("abcabcbb")==3); assert(s.lengthOfLongestSubstring("")==0);',
 ('数组','findAnagrams'): 'assert(s.findAnagrams("cbaebabacd","abc")==vector<int>({0,6})); assert(s.findAnagrams("ab","abc").empty());',
 ('数组','minWindow'): 'assert(s.minWindow("ADOBECODEBANC","ABC")=="BANC"); assert(s.minWindow("a","aa")=="");',
 ('数组','countSubarrays'): '''for(int mask=0;mask<243;++mask) {int z=mask;vector<int> a(5);for(int& x:a){x=z%3;z/=3;}
 for(int k=1;k<=3;++k) for(int m=1;m<=3;++m) {long long want=0;
 for(int l=0;l<5;++l) for(int r=l;r<5;++r) {map<int,int> cnt;for(int i=l;i<=r;++i) ++cnt[a[i]];bool ok=cnt.size()==(size_t)k;for(auto [x,c]:cnt) ok=ok && c>=m;want+=ok;}
 assert(s.countSubarrays(a,k,m)==want);}}''',
 ('数组','findSubstring'): 'vector<string> w{"foo","bar"}; auto r=s.findSubstring("barfoothefoobarman",w); sort(r.begin(),r.end()); assert(r==vector<int>({0,9})); w={"word","good","best","good"}; r=s.findSubstring("wordgoodgoodgoodbestword",w); assert(r==vector<int>({8}));',
 ('数组','subarraySum'): 'vector<int> a{1,1,1}; assert(s.subarraySum(a,2)==2); a={0,0,0}; assert(s.subarraySum(a,0)==6);',
 ('数组','productExceptSelf'): 'vector<int> a{1,2,3,4}; assert(s.productExceptSelf(a)==vector<int>({24,12,8,6})); a={-1,1,0,-3,3}; assert(s.productExceptSelf(a)==vector<int>({0,0,9,0,0}));',
 ('数组','search'): 'vector<int> a{1,3,5,7}; assert(s.search(a,5)==2); assert(s.search(a,2)==-1); a={4,5,6,7,0,1,2}; assert(s.search(a,0)==4);',
 ('数组','searchRange'): 'vector<int> a{5,7,7,8,8,10}; assert(s.searchRange(a,8)==vector<int>({3,4})); a={INT_MAX}; assert(s.searchRange(a,INT_MAX)==vector<int>({0,0}));',
 ('数组','findPeakElement'): 'vector<int> a{1,2,3,1}; assert(s.findPeakElement(a)==2); a={1}; assert(s.findPeakElement(a)==0);',
 ('数组','findMin'): 'vector<int> a{3,4,5,1,2}; assert(s.findMin(a)==1); a={1}; assert(s.findMin(a)==1);',
 ('数组','countElements'): 'vector<int> a{3,1,2}; assert(s.countElements(a,1)==2); a={5,5,5}; assert(s.countElements(a,1)==0);',
 ('数组','findMedianSortedArrays'): 'vector<int> a{1,3},b{2}; assert(s.findMedianSortedArrays(a,b)==2); a={}; b={1,2}; assert(s.findMedianSortedArrays(a,b)==1.5); a={INT_MAX}; b={INT_MAX}; assert(s.findMedianSortedArrays(a,b)==INT_MAX);',
 ('动态规划','fib'): 'assert(s.fib(0)==0); assert(s.fib(1)==1); assert(s.fib(10)==55);',
 ('动态规划','uniquePaths'): 'assert(s.uniquePaths(3,7)==28); assert(s.uniquePaths(1,1)==1);',
 ('动态规划','uniquePathsWithObstacles'): 'vector<vector<int>> a{{0,0,0},{0,1,0},{0,0,0}}; assert(s.uniquePathsWithObstacles(a)==2); a={{1}}; assert(s.uniquePathsWithObstacles(a)==0);',
 ('动态规划','minPathSum'): 'vector<vector<int>> a{{1,3,1},{1,5,1},{4,2,1}}; assert(s.minPathSum(a)==7);',
 ('动态规划','findTargetSumWays'): 'vector<int> a{1,1,1,1,1}; assert(s.findTargetSumWays(a,3)==5); a={0,0}; assert(s.findTargetSumWays(a,0)==4);',
 ('动态规划','canPartition'): 'vector<int> a{1,5,11,5}; assert(s.canPartition(a)); a={1,2,3,5}; assert(!s.canPartition(a));',
 ('动态规划','lengthOfLongestSubsequence'): 'vector<int> a{1,2,3,4,5}; assert(s.lengthOfLongestSubsequence(a,9)==3); a={1,1}; assert(s.lengthOfLongestSubsequence(a,3)==-1);',
 ('动态规划','findMaxForm'): 'vector<string> a{"10","0001","111001","1","0"}; assert(s.findMaxForm(a,5,3)==4);',
 ('动态规划','coinChange'): 'vector<int> a{1,2,5}; assert(s.coinChange(a,11)==3); a={2}; assert(s.coinChange(a,3)==-1); assert(s.coinChange(a,0)==0);',
 ('动态规划','numSquares'): 'assert(s.numSquares(12)==3); assert(s.numSquares(13)==2);',
 ('动态规划','change'): 'vector<int> a{1,2,5}; assert(s.change(5,a)==4); assert(s.change(0,a)==1);',
 ('动态规划','longestCommonSubsequence'): 'assert(s.longestCommonSubsequence("abcde","ace")==3); assert(s.longestCommonSubsequence("","a")==0);',
 ('动态规划','minDistance'): 'assert(s.minDistance("horse","ros")==3); assert(s.minDistance("","abc")==3); assert(s.minDistance("abc","")==3); assert(s.minDistance("intention","execution")==5);',
 ('动态规划','lengthOfLIS'): 'vector<int> a{10,9,2,5,3,7,101,18}; assert(s.lengthOfLIS(a)==4);',
 ('动态规划','maxSubArray'): 'vector<int> a{-2,1,-3,4,-1,2,1,-5,4}; assert(s.maxSubArray(a)==6); a={-3,-1,-2}; assert(s.maxSubArray(a)==-1);',
 ('动态规划','maxProduct'): 'vector<int> a{2,3,-2,4}; assert(s.maxProduct(a)==6); a={-2,0,-1}; assert(s.maxProduct(a)==0);',
 ('动态规划','maxSubarraySumCircular'): 'vector<int> a{5,-3,5}; assert(s.maxSubarraySumCircular(a)==10); a={-3,-2,-3}; assert(s.maxSubarraySumCircular(a)==-2);',
 ('动态规划','wordBreak'): 'vector<string> a{"leet","code"}; assert(s.wordBreak("leetcode",a)); a={"a"}; assert(!s.wordBreak("b",a));',
 ('动态规划','longestValidParentheses'): 'assert(s.longestValidParentheses(")()())")==4); assert(s.longestValidParentheses("")==0);',
 ('动态规划','isInterleave'): 'assert(s.isInterleave("aabcc","dbbca","aadbbcbcac")); assert(!s.isInterleave("aabcc","dbbca","aadbbbaccc")); assert(s.isInterleave("","",""));',
 ('动态规划','maxScore'): 'vector<int> a{-1,4,5,-2},b{-5,-1,-3,-2,-4}; assert(s.maxScore(a,b)==-1);',
 ('动态规划','numDistinct'): 'assert(s.numDistinct("rabbbit","rabbit")==3); assert(s.numDistinct("babgbag","bag")==5); assert(s.numDistinct("abc","")==1);',
 ('动态规划','longestPalindromeSubseq'): 'assert(s.longestPalindromeSubseq("bbbab")==4); assert(s.longestPalindromeSubseq("a")==1);',
 ('动态规划','minScoreTriangulation'): 'vector<int> a{1,3,1,4,1,5}; assert(s.minScoreTriangulation(a)==13);',
 ('动态规划','longestPalindrome'): 'auto r=s.longestPalindrome("babad"); assert(r=="bab" || r=="aba"); assert(s.longestPalindrome("cbbd")=="bb");',
 ('动态规划','maxCoins'): 'vector<int> a{3,1,5,8}; assert(s.maxCoins(a)==167);',
 ('动态规划','stoneGame'): 'vector<int> a{5,3,4,5}; assert(s.stoneGame(a));',
 ('动态规划','diameterOfBinaryTree'): 'TreeNode a(1),b(2),c(3),d(4),e(5); a.left=&b;a.right=&c;b.left=&d;b.right=&e; assert(s.diameterOfBinaryTree(&a)==3);',
 ('动态规划','maxPathSum'): 'TreeNode a(-3); assert(s.maxPathSum(&a)==-3); TreeNode b(1),c(2),d(3);b.left=&c;b.right=&d;assert(s.maxPathSum(&b)==6);',
 ('动态规划','longestPath'): 'vector<int> a{-1,0,0,1,1,2}; assert(s.longestPath(a,"abacbe")==3);',
 ('动态规划','minCameraCover'): 'TreeNode a(0),b(0),c(0),d(0);a.left=&b;b.left=&c;b.right=&d; assert(s.minCameraCover(&a)==1); TreeNode e(0); assert(s.minCameraCover(&e)==1);',
 ('图','numIslands'): 'vector<vector<char>> a{{\'1\',\'1\',\'0\'},{\'0\',\'1\',\'0\'},{\'0\',\'0\',\'1\'}}; assert(s.numIslands(a)==2);',
 ('图','orangesRotting'): 'vector<vector<int>> a{{2,1,1},{1,1,0},{0,1,1}}; assert(s.orangesRotting(a)==4); a={{0}}; assert(s.orangesRotting(a)==0); a={{1}}; assert(s.orangesRotting(a)==-1);',
 ('图','minMutation'): 'vector<string> a{"AACCGGTA"}; assert(s.minMutation("AACCGGTT","AACCGGTA",a)==1);',
 ('图','ladderLength'): 'vector<string> a{"hot","dot","dog","lot","log","cog"}; assert(s.ladderLength("hit","cog",a)==5);',
 ('图','canFinish'): 'vector<vector<int>> a{{1,0}}; assert(s.canFinish(2,a)); a={{1,0},{0,1}}; assert(!s.canFinish(2,a));',
 ('图','findOrder'): 'vector<vector<int>> a{{1,0}}; assert(s.findOrder(2,a)==vector<int>({0,1})); a={{1,0},{0,1}}; assert(s.findOrder(2,a).empty());',
 ('贪心','canJump'): 'vector<int> a{2,3,1,1,4}; assert(s.canJump(a)); a={3,2,1,0,4}; assert(!s.canJump(a));',
 ('贪心','jump'): 'vector<int> a{2,3,1,1,4}; assert(s.jump(a)==2); a={0}; assert(s.jump(a)==0);',
 ('贪心','partitionLabels'): 'assert(s.partitionLabels("ababcbacadefegdehijhklij")==vector<int>({9,7,8}));',
 ('位运算','singleNumber'): 'vector<int> a{4,1,2,1,2}; assert(s.singleNumber(a)==4);',
 ('位运算','rangeBitwiseAnd'): 'assert(s.rangeBitwiseAnd(5,7)==4); assert(s.rangeBitwiseAnd(0,INT_MAX)==0);',
 ('数据结构','findKthLargest'): 'vector<int> a{3,2,3,1,2,4,5,5,6}; assert(s.findKthLargest(a,4)==4); a={2,2,2}; assert(s.findKthLargest(a,2)==2);',
 ('数据结构','kSmallestPairs'): 'vector<int> a{1,7,11},b{2,4,6}; auto r=s.kSmallestPairs(a,b,3); assert(r==vector<vector<int>>({{1,2},{1,4},{1,6}})); a={1}; b={2}; assert(s.kSmallestPairs(a,b,5).size()==1);',
 ('数据结构','dailyTemperatures'): 'vector<int> a{73,74,75,71,69,72,76,73}; assert(s.dailyTemperatures(a)==vector<int>({1,1,4,2,1,1,0,0}));',
 ('数据结构','trap'): 'vector<int> a{0,1,0,2,1,0,1,3,2,1,2,1}; assert(s.trap(a)==6);',
 ('数据结构','largestRectangleArea'): 'vector<int> a{2,1,5,6,2,3}; assert(s.largestRectangleArea(a)==10); a={2,2}; assert(s.largestRectangleArea(a)==4);',
 ('数据结构','maxSlidingWindow'): 'vector<int> a{1,3,-1,-3,5,3,6,7}; assert(s.maxSlidingWindow(a,3)==vector<int>({3,3,5,5,6,7})); a={1,1,1}; assert(s.maxSlidingWindow(a,2)==vector<int>({1,1}));',
 ('数据结构','findWords'): 'vector<vector<char>> a{{\'o\',\'a\',\'a\',\'n\'},{\'e\',\'t\',\'a\',\'e\'},{\'i\',\'h\',\'k\',\'r\'},{\'i\',\'f\',\'l\',\'v\'}}; auto original=a; vector<string> w{"oath","pea","eat","rain"}; auto r=s.findWords(a,w); sort(r.begin(),r.end()); assert(r==vector<string>({"eat","oath"})); assert(a==original);',
 ('数据结构','numOfUnplacedFruits'): 'vector<int> a{4,2,5},b{3,5,4}; assert(s.numOfUnplacedFruits(a,b)==1);',
 ('回溯','combinationSum'): 'vector<int> a{2,3,6,7}; auto r=s.combinationSum(a,7); sort(r.begin(),r.end()); assert(r==vector<vector<int>>({{2,2,3},{7}}));',
 ('回溯','permute'): 'vector<int> a{1,2,3},original=a; auto r=s.permute(a); sort(r.begin(),r.end()); assert(r.size()==6); assert(unique(r.begin(),r.end())==r.end()); assert(a==original);',
 ('二叉树','rightSideView'): 'TreeNode a(1),b(2),c(3),d(5);a.left=&b;a.right=&c;b.right=&d;assert(s.rightSideView(&a)==vector<int>({1,3,5}));',
 ('二叉树','pathSum'): 'TreeNode a(10),b(5),c(-3),d(3),e(2),f(11),g(3),h(-2),i(1);a.left=&b;a.right=&c;b.left=&d;b.right=&e;c.right=&f;d.left=&g;d.right=&h;e.right=&i;assert(s.pathSum(&a,8)==3);',
 ('二叉树','widthOfBinaryTree'): 'TreeNode a(1),b(3),c(2),d(5),e(3),f(9);a.left=&b;a.right=&c;b.left=&d;b.right=&e;c.right=&f;assert(s.widthOfBinaryTree(&a)==4); vector<TreeNode> chain(100);for(int i=0;i<99;++i) chain[i].right=&chain[i+1];assert(s.widthOfBinaryTree(&chain[0])==1);',
 ('链表','reverseList'): 'ListNode a(1),b(2),c(3);a.next=&b;b.next=&c; assert(s.reverseList(&a)==&c); assert(c.next==&b && b.next==&a && !a.next);',
 ('链表','reverseKGroup'): 'ListNode a(1),b(2),c(3),d(4),e(5);a.next=&b;b.next=&c;c.next=&d;d.next=&e;assert(s.reverseKGroup(&a,2)==&b);assert(b.next==&a && a.next==&d && d.next==&c && c.next==&e && !e.next);',
 ('链表','isPalindrome'): 'ListNode a(1),b(2),c(2),d(1);a.next=&b;b.next=&c;c.next=&d;assert(s.isPalindrome(&a));d.val=3;assert(!s.isPalindrome(&a));',
 ('链表','sortList'): 'ListNode a(4),b(2),c(1),d(3);a.next=&b;b.next=&c;c.next=&d;auto r=s.sortList(&a);assert(r==&c && c.next==&b && b.next==&d && d.next==&a && !a.next);',
}

def main():
    snippets = list(examples())
    units = []
    runs = []
    for path, index, code in snippets:
        topic = path.parent.name
        test = None
        for (category, method), case in CASES.items():
            if category == topic and re.search(r'\b'+method+r'\(', code):
                test = case
                # Ordinary binary-search examples require sorted input.
                if topic == '数组' and method == 'search' and 'is_blue' not in code:
                    test = 'vector<int> a{1,3,5,7}; assert(s.search(a,5)==2); assert(s.search(a,2)==-1); a={};assert(s.search(a,0)==-1);'
                break
        if topic == '位运算' and 'cnt1 % 3' in code:
            test = 'vector<int> a{2,2,3,2};assert(s.singleNumber(a)==3);a={-2,-2,-2,-3};assert(s.singleNumber(a)==-3);'
        if 'zero_one_knapsack(' in code:
            test = 'vector<int> w{1,3,4},v{15,20,30};assert(zero_one_knapsack(4,w,v)==35);'
        if 'unbounded_knapsack(' in code:
            test = 'vector<int> w{1,3,4},v{15,20,30};assert(unbounded_knapsack(4,w,v)==60);'
        if 'class LRUCache' in code:
            test = 'LRUCache l(2);l.put(1,1);l.put(2,2);assert(l.get(1)==1);l.put(3,3);assert(l.get(2)==-1);l.put(1,10);assert(l.get(1)==10);l.put(4,4);assert(l.get(3)==-1);'
        if 'class Trie {' in code:
            test = 'Trie t;t.insert("apple");assert(t.search("apple"));assert(!t.search("app"));assert(t.startsWith("app"));t.insert("app");assert(t.search("app"));'
        if 'class NumArray' in code:
            test = 'vector<int> a{1,3,5};NumArray n(a);assert(n.sumRange(0,2)==9);n.update(1,2);assert(n.sumRange(0,2)==8);assert(n.sumRange(1,1)==2);'
        if 'int rob(vector<int>' in code:
            # Circular and linear variants share these particular expected answers.
            test = 'vector<int> a{1,2,3,1};assert(s.rob(a)==4);a={1};assert(s.rob(a)==1);'
        if 'int rob(TreeNode' in code:
            test = 'TreeNode a(3),b(2),c(3),d(3),e(1);a.left=&b;a.right=&c;b.right=&d;c.right=&e;assert(s.rob(&a)==7);'
        if 'int mergeStones(' in code:
            test = 'vector<int> a{3,2,4,1};assert(s.mergeStones(a,2)==20);assert(s.mergeStones(a,3)==-1);' if 'int k)' in code else 'vector<int> a{3,2,4,1};assert(s.mergeStones(a)==20);'
        if 'maxProfit(' in code:
            if 'int k,' in code:
                if '不可达' in code:
                    # Exact / at-least counts: this data admits two profitable transactions.
                    test = 'vector<int> a{3,2,6,5,0,3};assert(s.maxProfit(2,a)==7);'
                else:
                    test = 'vector<int> a{3,2,6,5,0,3};assert(s.maxProfit(2,a)==7);assert(s.maxProfit(0,a)==0);'
            elif 'dp[i - 2]' in code:
                test = 'vector<int> a{1,2,3,0,2};assert(s.maxProfit(a)==3);'
            else:
                test = 'vector<int> a{7,1,5,3,6,4};assert(s.maxProfit(a)==7);'
        if not test:
            continue
        name = f'example_{len(units)}'
        setup = 'Solution s;' if 'class Solution' in code else ''
        units.append(f'namespace {name} {{\n{code}\nvoid run() {{ {setup} {test} }}\n}}')
        runs.append(f'{name}::run();')
    program = HEADERS + '\n'.join(units) + '\nint main(){\n' + '\n'.join(runs) + f'\ncout << "{len(units)} documented variants passed behavior checks\\n";\n}}'
    with tempfile.TemporaryDirectory(prefix='code-notes-tests-') as temp:
        source, binary = Path(temp)/'examples.cpp', Path(temp)/'examples'
        source.write_text(program)
        subprocess.run(['clang++','-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(source),'-o',str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    main()
