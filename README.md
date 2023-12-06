# OpenSpeech: 中文开源开放共建数据集

在语音识别领域，中文的开源数据集太少了。虽然[WenetSpeech](https://github.com/wenet-e2e/wenetspeech)的开源把这一量级推上了万小时级别，但相比英文数据动辄数万小时的开源数据还是有不小的差距，在目前大数据大模型的发展趋势下更显不足。本项目旨在构建一个数万小时以上的用于学术研究的中文开源数据集。

## 技术方案

k2 团队前段时间开源了5万小时的英文数据集[Libriheavy](https://github.com/k2-fsa/libriheavy), 并提供了一个数据集对齐的工具包[textsearch](https://github.com/k2-fsa/text_search), 本项目拟从网络收集诸如有声书、影视剧、视频等音频资源，并找到相应的文本（有声书通过下载电子书，视频通过 OCR 技术），然后使用 textsearch 对齐音频和文本得到最终的数据集。搜集数据是个费时费力的过程，欢迎更多有兴趣有热情的同学一起参与共建。


## 数据源

目前我们已经收集了一些数据，正在进行对齐和整理，具体数据源如下：

* [电视剧](source/tv.md)
* [有声书](source/audio_book.md)
* [综艺及演讲等](source/talk_show.md)
* [电影纪录片等](source/movie.md)


## 授权形式

本数据集将只授权给学术研究使用，拟采用 CC BY-NC-ND 4.0 授权协议。通过网络爬取的数据多少会涉及到版权等问题，我们将对数据进行必要的切分和打乱，避免再分发大段原始音频。如本数据集侵犯到您的权益，也请告知我们，我们会第一时间删除对应的数据。
