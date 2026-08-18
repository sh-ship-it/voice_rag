import { useState } from 'react';

interface Chunk {
  chunk_id: string;
  text: string;
  language?: string;
}

interface Props {
  citations: string[];
  chunks?: Chunk[];
  loading?: boolean;
}

export function CitationAccordion({ citations, chunks, loading }: Props) {
  const [openIds, setOpenIds] = useState<Set<string>>(new Set(citations ? citations.slice(0, 1) : []));

  const hasCitations = citations && citations.length > 0;
  const chunkMap = new Map((chunks || []).map((c) => [c.chunk_id, c]));

  const toggle = (id: string) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="nb-card citation-container">
      <div className="citation-container__header">
        <span className="nb-badge nb-badge--dark">
          📚 GROUNDED CITATIONS ({hasCitations ? citations.length : 0})
        </span>
        <span className="citation-container__hint">MSMARCO-XI Corpus</span>
      </div>

      <div className="citation-list">
        {loading && !hasCitations ? (
          <div className="citation-empty-note">
            <span className="inline-spinner"></span> Retrieving top supporting passage citations...
          </div>
        ) : !hasCitations ? (
          <div className="citation-empty-note">
            No active citations. Retrieved passage cards with exact IDs will appear here.
          </div>
        ) : (
          citations.map((cid, i) => {
            const chunk = chunkMap.get(cid);
            const isOpen = openIds.has(cid);
            return (
              <div key={cid} className={`citation-item ${isOpen ? 'citation-item--open' : ''}`}>
                <button
                  type="button"
                  className="citation-item__btn"
                  onClick={() => toggle(cid)}
                  aria-expanded={isOpen}
                  id={`citation-btn-${i}`}
                >
                  <div className="citation-item__btn-left">
                    <span className="nb-badge nb-badge--light">[{i + 1}]</span>
                    <span className="citation-item__id">{cid}</span>
                  </div>
                  <span className="citation-item__arrow">{isOpen ? '▲ CLOSE' : '▼ VIEW PASSAGE'}</span>
                </button>

                {isOpen && (
                  <div className="citation-item__body" role="region" aria-labelledby={`citation-btn-${i}`}>
                    {chunk ? (
                      <p className="citation-item__text" dir="auto">
                        "{chunk.text}"
                      </p>
                    ) : (
                      <p className="citation-item__text citation-item__text--missing">
                        Passage content retrieved for {cid}.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
