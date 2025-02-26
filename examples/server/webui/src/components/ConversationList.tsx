import { useEffect, useState } from 'react';
import { classNames } from '../utils/misc';
import { Conversation } from '../utils/types';
import StorageUtils from '../utils/storage';
import { useNavigate, useParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { useAppContext } from '../utils/app.context.tsx';

export function ConversationListDownloadDeleteButtonHeader({
  classAdd,
}: {
  classAdd: string;
}) {
  const { t } = useTranslation();
  const { isGenerating, viewingChat } = useAppContext();
  const isCurrConvGenerating = isGenerating(viewingChat?.conv.id ?? '');
  const navigate = useNavigate();

  const removeConversation = () => {
    if (isCurrConvGenerating || !viewingChat) return;
    const convId = viewingChat?.conv.id;
    if (window.confirm(t('ConversationList.deleteConfirm'))) {
      StorageUtils.remove(convId);
      navigate('/');
    }
  };

  const downloadConversation = () => {
    if (isCurrConvGenerating || !viewingChat) return;
    const convId = viewingChat?.conv.id;
    const conversationJson = JSON.stringify(viewingChat, null, 2);
    const blob = new Blob([conversationJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${convId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      {viewingChat && (
        <>
          <div
            className={classAdd + ' tooltip tooltip-bottom z-100'}
            data-tip={t('ConversationList.newConversation')}
            onClick={() => {
              navigate('/');
              const elem = document.getElementById(
                'toggle-conversation-list'
              ) as HTMLInputElement;
              if (elem && elem.checked) {
                elem.click();
              }
            }}
          >
            <button
              role="button"
              className={classNames({
                'btn m-1 ': true,
              })}
              aria-label={t('ConversationList.newConversation')}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="size-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v6m3-3H9m12 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                />
              </svg>
            </button>
          </div>
          <div
            className={classAdd + ' tooltip tooltip-bottom z-100'}
            data-tip={t('ConversationList.downloadBtn')}
            onClick={downloadConversation}
          >
            <button
              role="button"
              className="btn m-1"
              disabled={isCurrConvGenerating}
              aria-label={t('ConversationList.downloadBtn')}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="size-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0-3-3m3 3 3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z"
                />
              </svg>
            </button>
          </div>
          <div
            className={classAdd + ' tooltip tooltip-bottom z-100'}
            data-tip={t('ConversationList.deleteBtn')}
            onClick={removeConversation}
          >
            <button
              role="button"
              className="btn m-1"
              disabled={isCurrConvGenerating}
              aria-label={t('ConversationList.deleteBtn')}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="size-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="m20.25 7.5-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5m6 4.125 2.25 2.25m0 0 2.25 2.25M12 13.875l2.25-2.25M12 13.875l-2.25 2.25M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z"
                />
              </svg>
            </button>
          </div>
        </>
      )}
    </>
  );
}

export function ConversationListButton() {
  const { t } = useTranslation();
  return (
    <>
      {/* open sidebar button */}
      <div
        className="tooltip tooltip-bottom z-100"
        data-tip={t('ConversationList.conversationBtn')}
        onClick={() => {
          const elem = document.getElementById('convBlock');
          if (elem) {
            elem.style.display = 'block';
          }
        }}
      >
        <label htmlFor="toggle-conversation-list" className="btn m-1 lg:hidden">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="size-6"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 0 1-.825-.242m9.345-8.334a2.126 2.126 0 0 0-.476-.095 48.64 48.64 0 0 0-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0 0 11.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155"
            />
          </svg>
        </label>
      </div>
      <ConversationListDownloadDeleteButtonHeader classAdd="hidden sm:block lg:hidden" />
    </>
  );
}

export default function ConversationList() {
  const { t } = useTranslation();
  const params = useParams();
  const navigate = useNavigate();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currConv, setCurrConv] = useState<Conversation | null>(null);

  useEffect(() => {
    StorageUtils.getOneConversation(params.convId ?? '').then(setCurrConv);
  }, [params.convId]);

  useEffect(() => {
    const handleConversationChange = async () => {
      setConversations(await StorageUtils.getAllConversations());
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    handleConversationChange();
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
  }, []);

  return (
    <>
      <div className="h-full flex flex-col max-w-64 py-4 px-4">
        <div className="flex flex-row items-center justify-between mb-4 mt-4">
          <h2 className="font-bold ml-4">
            {t('ConversationList.Conversations')}
          </h2>
          {/* close sidebar button */}

          <div
            className="tooltip tooltip-bottom z-100"
            data-tip={t('ConversationList.closeBtn')}
            onClick={() => {
              const elem = document.getElementById('convBlock');
              if (elem) {
                if (elem.style.display === 'none') {
                  elem.style.display = 'block';
                } else {
                  elem.style.display = 'none';
                }
              }
            }}
          >
            <label className="btn btn-ghost m-1 lg:hidden">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="size-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                />
              </svg>
            </label>
          </div>
        </div>
        <div className="w-full sm:hidden lg:block">
          <div className="flex flex-col items-center">
            <span>
              <ConversationListDownloadDeleteButtonHeader classAdd="" />
            </span>
          </div>
        </div>
        {/* list of conversations */}
        {conversations.map((conv) => (
          <div
            key={conv.id}
            className={classNames({
              'btn btn-ghost justify-start font-normal': true,
              'btn-active': conv.id === currConv?.id,
            })}
            onClick={() => {
              navigate(`/chat/${conv.id}`);
              const elem = document.getElementById(
                'toggle-conversation-list'
              ) as HTMLInputElement;
              if (elem && elem.checked) {
                elem.click();
              }
            }}
            dir="auto"
          >
            <span className="truncate">{conv.name}</span>
          </div>
        ))}
        <div className="text-center text-xs opacity-40 mt-auto mx-4">
          {t('ConversationList.convInformation')}
        </div>
      </div>
    </>
  );
}
